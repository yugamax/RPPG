import os
import glob
import numpy as np

os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

import keras
from keras import layers, mixed_precision
import cv2
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import warnings
from typing import Any, cast
warnings.filterwarnings("ignore")

# ── Try MediaPipe; fall back gracefully ───────────────────────────────────────
try:
    import mediapipe as mp
    mp_face_detection = getattr(mp.solutions, "face_detection")
    _MP_FACE = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    HAVE_MEDIAPIPE = True
    print("MediaPipe face detection available — using as primary detector.")
except Exception:
    _MP_FACE = None
    HAVE_MEDIAPIPE = False
    print("MediaPipe not found — using Haar cascade only.")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    DATA_DIR        = "data/UBFC_DATASET/DATASET_1"
    CLIP_LEN        = 96
    IMG_H           = 64
    IMG_W           = 64
    BATCH_SIZE      = 2
    EPOCHS          = 40
    LR              = 3e-4
    VAL_SPLIT       = 0.2          # fraction of SUBJECTS for validation (not clips)
    SEED            = 42
    FPS             = 30.0
    BPM_LOW         = 40
    BPM_HIGH        = 240
    CHECKPOINT_DIR  = "checkpoints"
    LOG_DIR         = "logs"
    FACE_CASCADE: str | None = None
    GT_FILENAME     = "gtdump.xmp"
    STRIDE          = 48
    AUGMENT         = True
    MIXED_PRECISION = True
    ROI_SMOOTH_ALPHA = 0.25        # EMA smoothing for face ROI (0=frozen, 1=raw)


def _face_cascade_path() -> str:
    return os.path.join(getattr(cv2, "data").haarcascades,
                        "haarcascade_frontalface_default.xml")


cfg = Config()
cfg.FACE_CASCADE = _face_cascade_path()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)

if cfg.MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled (fp16)")


# ─────────────────────────────────────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(sig: np.ndarray, fs: float,
                    low: float = 0.7, high: float = 4.0) -> np.ndarray:
    nyq = 0.5 * fs
    sos = sp_signal.butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, sig)


def signal_to_bpm(sig: np.ndarray, fs: float) -> float:
    freqs    = rfftfreq(len(sig), d=1.0 / fs)
    spectrum = np.abs(rfft(sig - np.mean(sig)))
    mask     = (freqs >= cfg.BPM_LOW / 60.0) & (freqs <= cfg.BPM_HIGH / 60.0)
    if not np.any(mask):
        return 0.0
    return freqs[mask][np.argmax(spectrum[mask])] * 60.0


def estimate_spo2(red_sig: np.ndarray,
                  ir_sig: np.ndarray | None = None) -> float:
    if ir_sig is not None and len(ir_sig) > 0:
        r = (np.std(red_sig) / (np.abs(np.mean(red_sig)) + 1e-8)) / \
            (np.std(ir_sig)  / (np.abs(np.mean(ir_sig))  + 1e-8))
    else:
        r = np.std(red_sig) / (np.abs(np.mean(red_sig)) + 1e-8)
    return float(np.clip(110.0 - 25.0 * r, 70.0, 100.0))


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation."""
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 8 — Signal alignment: resample GT PPG to match exact video frame count
# ─────────────────────────────────────────────────────────────────────────────

def resample_ppg_to_frames(ppg: np.ndarray, n_video_frames: int) -> np.ndarray:
    """
    Resample GT PPG so its length exactly matches the number of video frames.
    This handles UBFC subjects where GT is sampled at a slightly different rate.
    """
    n_ppg = len(ppg)
    if n_ppg == n_video_frames:
        return ppg
    old_t = np.linspace(0.0, 1.0, n_ppg)
    new_t = np.linspace(0.0, 1.0, n_video_frames)
    return interp1d(
        old_t,
        ppg,
        kind="linear",
    )(new_t).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_ground_truth(gt_path: str) -> np.ndarray:
    """
    UBFC gtdump.xmp: rows are [time, HR, SpO2, ppg, ...].
    Pick the column with highest coefficient of variation (most likely raw PPG).
    """
    try:
        data = np.loadtxt(gt_path, delimiter=',')
    except Exception:
        data = np.loadtxt(gt_path)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[1] == 1:
        return data[:, 0].astype(np.float32)

    cv = [np.std(data[:, c]) / (np.abs(np.mean(data[:, c])) + 1e-8)
          for c in range(data.shape[1])]
    return data[:, int(np.argmax(cv))].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 6 + 11 — Stable Face ROI: MediaPipe primary, Haar fallback, EMA smoothing
# ─────────────────────────────────────────────────────────────────────────────

class FaceROIExtractor:
    def __init__(self):
        self._haar        = cv2.CascadeClassifier(cfg.FACE_CASCADE or "")
        self._ema_box     = None          # smoothed (x,y,w,h) as floats
        self._miss_count  = 0
        self._total       = 0
        self._alpha       = cfg.ROI_SMOOTH_ALPHA

    # ── detection ─────────────────────────────────────────────────────────────
    def _detect(self, frame: np.ndarray):
        """Return (x,y,w,h) of largest face, or None."""
        h_frame, w_frame = frame.shape[:2]

        # 1. MediaPipe (more robust, handles side-on poses)
        if HAVE_MEDIAPIPE and _MP_FACE is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = _MP_FACE.process(rgb)
            if res.detections:
                best = max(res.detections,
                           key=lambda d: d.location_data.relative_bounding_box.width *
                                         d.location_data.relative_bounding_box.height)
                bb = best.location_data.relative_bounding_box
                x  = int(bb.xmin * w_frame)
                y  = int(bb.ymin * h_frame)
                w  = int(bb.width  * w_frame)
                h  = int(bb.height * h_frame)
                return (x, y, w, h)

        # 2. Haar cascade fallback
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            return tuple(max(faces, key=lambda r: r[2] * r[3]))

        return None

    # ── EMA smoothing ─────────────────────────────────────────────────────────
    def _update_ema(self, box):
        """Exponential moving average on the bounding box."""
        bx = np.array(box, dtype=np.float32)
        if self._ema_box is None:
            self._ema_box = bx
        else:
            alpha = np.float32(self._alpha)
            self._ema_box = alpha * bx + (np.float32(1.0) - alpha) * self._ema_box

    # ── public API ────────────────────────────────────────────────────────────
    def extract(self, frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        self._total += 1
        det = self._detect(frame)

        if det is not None:
            x, y, w, h = det
            # Expand upward to include forehead
            pad_y = int(h * 0.10)
            y = max(0, y - pad_y)
            h = min(frame.shape[0] - y, h + pad_y)
            self._update_ema((x, y, w, h))
        else:
            self._miss_count += 1

        if self._ema_box is not None:
            x, y, w, h = (int(v) for v in self._ema_box)
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            roi = frame[y: y + h, x: x + w]
        else:
            roi = frame

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return cv2.resize(roi_rgb, (out_w, out_h)).astype(np.float32) / 255.0

    def face_miss_rate(self) -> float:
        return self._miss_count / max(self._total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_clip(clip: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        clip = clip[:, :, ::-1, :]                        # horizontal flip
    alpha = np.random.uniform(0.85, 1.15)
    beta  = np.random.uniform(-0.05, 0.05)
    clip  = np.clip(clip * alpha + beta, 0.0, 1.0)        # brightness/contrast
    # NOTE: temporal reversal removed — it reverses GT PPG alignment too
    return clip.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 10 — Sanity check: plot RGB mean and GT PPG for first subject
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(video_path: str, gt_ppg: np.ndarray,
                 out_path: str, n_frames: int = 300):
    """
    Extracts mean-RGB signal from the first n_frames of a video and plots
    it alongside the GT PPG. A good alignment shows correlated oscillations.
    """
    cap       = cv2.VideoCapture(video_path)
    extractor = FaceROIExtractor()
    r_vals, g_vals, b_vals = [], [], []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        roi = extractor.extract(frame, cfg.IMG_H, cfg.IMG_W)
        r_vals.append(roi[:, :, 0].mean())
        g_vals.append(roi[:, :, 1].mean())
        b_vals.append(roi[:, :, 2].mean())
    cap.release()

    n = min(len(r_vals), len(gt_ppg))
    t = np.arange(n) / cfg.FPS

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t, r_vals[:n], color="tomato",   label="R channel mean", linewidth=0.8)
    axes[0].plot(t, g_vals[:n], color="seagreen", label="G channel mean", linewidth=0.8)
    axes[0].plot(t, b_vals[:n], color="steelblue",label="B channel mean", linewidth=0.8)
    axes[0].set_title("Face ROI mean-RGB (sanity check)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, gt_ppg[:n], color="purple", linewidth=0.8)
    axes[1].set_title("Ground-truth PPG")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Sanity check saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Video Loading  (FIX 2 + 8 applied here)
# ─────────────────────────────────────────────────────────────────────────────

def load_video_clips(video_path: str, gt_ppg: np.ndarray,
                     clip_len: int, stride: int):
    cap       = cv2.VideoCapture(video_path)
    extractor = FaceROIExtractor()
    frames    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extractor.extract(frame, cfg.IMG_H, cfg.IMG_W))
    cap.release()

    miss_rate = extractor.face_miss_rate()
    if miss_rate > 0.30:
        print(f"  [WARN] face miss rate {miss_rate:.0%} — ROI quality may be poor")

    frames = np.array(frames, dtype=np.float32)
    n_vid  = len(frames)

    # FIX 8: resample GT PPG to match exact video frame count
    gt_ppg = resample_ppg_to_frames(gt_ppg, n_vid)

    # FIX 2: normalise GT PPG globally across the whole recording BEFORE slicing.
    #        This prevents each clip having a different amplitude scale.
    gt_ppg_norm = normalize_signal(bandpass_filter(gt_ppg, cfg.FPS))

    clips, targets = [], []
    for start in range(0, n_vid - clip_len + 1, stride):
        end = start + clip_len
        clips.append(frames[start:end])
        targets.append(gt_ppg_norm[start:end])

    if not clips:
        return None, None

    return np.array(clips, dtype=np.float32), np.array(targets, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset I/O  (FIX 3 — subject-level split metadata stored alongside data)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(data_dir: str) -> str:
    tag = f"cl{cfg.CLIP_LEN}_s{cfg.STRIDE}_h{cfg.IMG_H}_w{cfg.IMG_W}"
    return os.path.join(data_dir, f"dataset_{tag}.npz")


def save_dataset(X: np.ndarray, y: np.ndarray,
                 subj_ids: np.ndarray, data_dir: str):
    path = _cache_path(data_dir)
    np.savez(path, X=X, y=y, subj_ids=subj_ids)
    print(f"Dataset saved to {os.path.basename(path)}")


def load_dataset(data_dir: str):
    path = _cache_path(data_dir)
    if os.path.exists(path):
        data = np.load(path)
        subj_ids = data['subj_ids'] if 'subj_ids' in data else None
        print(f"Loaded dataset from cache: {os.path.basename(path)}")
        return data['X'], data['y'], subj_ids
    old_path = os.path.join(data_dir, "dataset.npz")
    if os.path.exists(old_path):
        print("  [WARNING] Stale cache (dataset.npz) found — ignoring, re-extracting.")
    return None, None, None


def build_dataset(data_dir: str):
    X, y, subj_ids = load_dataset(data_dir)
    if X is not None and y is not None and subj_ids is not None:
        return X, y, subj_ids

    subject_dirs = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(p)
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No subjects found under {data_dir}")

    all_clips, all_targets, all_subj_ids = [], [], []
    sanity_done = False

    for idx, subj in enumerate(subject_dirs):
        vid_path = os.path.join(subj, "vid.avi")
        gt_path  = os.path.join(subj, cfg.GT_FILENAME)

        if not (os.path.exists(vid_path) and os.path.exists(gt_path)):
            print(f"  [SKIP] {subj} — missing files")
            continue

        print(f"  Loading {os.path.basename(subj)} …", end=" ", flush=True)
        gt_ppg = parse_ground_truth(gt_path)

        # FIX 10: sanity check on first subject only
        if not sanity_done:
            sanity_path = os.path.join(cfg.LOG_DIR, "sanity_check.png")
            sanity_check(vid_path, gt_ppg, sanity_path)
            sanity_done = True

        clips, targets = load_video_clips(vid_path, gt_ppg, cfg.CLIP_LEN, cfg.STRIDE)

        if clips is None:
            print("no clips extracted")
            continue

        all_clips.append(clips)
        all_targets.append(targets)
        # Tag every clip with its subject index (for leakage-free splitting)
        all_subj_ids.append(np.full(len(clips), idx, dtype=np.int32))
        print(f"{len(clips)} clips")

    X        = np.concatenate(all_clips,    axis=0)
    y        = np.concatenate(all_targets,  axis=0)
    subj_ids = np.concatenate(all_subj_ids, axis=0)
    print(f"\nTotal: {X.shape[0]} clips | X {X.shape} | y {y.shape}")
    save_dataset(X, y, subj_ids, data_dir)
    return X, y, subj_ids


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Subject-level train / val split (no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def subject_split(X: np.ndarray, y: np.ndarray, subj_ids: np.ndarray,
                  val_fraction: float = cfg.VAL_SPLIT, seed: int = cfg.SEED):
    """
    Split by unique subject, not by clip.
    All clips from a given subject go entirely to train OR val — never both.
    """
    rng          = np.random.default_rng(seed)
    unique_subjs = np.unique(subj_ids)
    rng.shuffle(unique_subjs)

    n_val        = max(1, int(len(unique_subjs) * val_fraction))
    val_subjs    = set(unique_subjs[:n_val].tolist())
    train_subjs  = set(unique_subjs[n_val:].tolist())

    train_mask = np.isin(subj_ids, list(train_subjs))
    val_mask   = np.isin(subj_ids, list(val_subjs))

    print(f"  Subject split — train subjects: {sorted(train_subjs)} | "
          f"val subjects: {sorted(val_subjs)}")
    print(f"  Train clips: {train_mask.sum()} | Val clips: {val_mask.sum()}")

    return (X[train_mask], y[train_mask],
            X[val_mask],   y[val_mask])


# ─────────────────────────────────────────────────────────────────────────────
# tf.data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def augment_tf(clip, target):
    clip = tf.numpy_function(
        lambda c: augment_clip(c), [clip], tf.float32
    )
    clip.set_shape([cfg.CLIP_LEN, cfg.IMG_H, cfg.IMG_W, 3])
    return clip, target


def make_tf_dataset(X: np.ndarray, y: np.ndarray,
                    training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=cfg.SEED)
        if cfg.AUGMENT:
            ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# Model — PhysNet v2 with residual 3D blocks + temporal attention
# ─────────────────────────────────────────────────────────────────────────────

def conv_bn_relu(x, filters, kernel, padding="same", strides=(1, 1, 1)):
    x = layers.Conv3D(filters, kernel, strides=strides,
                      padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def residual_3d_block(x, filters):
    shortcut = x
    x = conv_bn_relu(x, filters, (3, 3, 3))
    x = layers.Conv3D(filters, (3, 3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, (1, 1, 1), padding="same",
                                 use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def temporal_attention(x):
    C        = x.shape[-1]
    head_dim = min(C // 4, 32)
    q = layers.Dense(head_dim, use_bias=False)(x)
    k = layers.Dense(head_dim, use_bias=False)(x)
    v = layers.Dense(C,        use_bias=False)(x)
    scores  = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
                  tf.cast(head_dim, tf.float32))
    weights = tf.nn.softmax(scores, axis=-1)
    out     = tf.matmul(weights, v)
    x       = layers.Add()([x, out])
    return layers.LayerNormalization()(x)


def build_physnet(clip_len: int, img_h: int, img_w: int) -> keras.Model:
    inp = keras.Input(shape=(clip_len, img_h, img_w, 3), name="video_clip")

    x = conv_bn_relu(inp, 32, (1, 5, 5))
    x = conv_bn_relu(x,  32, (3, 3, 3))
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = residual_3d_block(x, 64)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = residual_3d_block(x, 128)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = residual_3d_block(x, 256)
    x = layers.MaxPool3D((2, 2, 2))(x)

    x = residual_3d_block(x, 256)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=[2, 3]),
                      name="spatial_avg")(x)

    x = layers.Lambda(lambda t: temporal_attention(t),
                      name="temporal_attn")(x)

    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64,  3, padding="same", activation="relu")(x)

    # Upsample temporal dim back to clip_len
    x = layers.Lambda(lambda t: tf.image.resize(
        tf.expand_dims(t, -1), [clip_len, 1], method="bilinear"))(x)
    x = layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(x)

    x   = layers.Conv1D(1, 1, padding="same")(x)
    out = layers.Lambda(lambda t: tf.squeeze(t, axis=-1),
                        name="rppg_signal")(x)

    return keras.Model(inp, out, name="PhysNet_v2")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 9 — Loss: drop MSE, use Pearson + frequency only (both amplitude-invariant)
# ─────────────────────────────────────────────────────────────────────────────

def pearson_loss(y_true, y_pred):
    yt  = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    yp  = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    num = tf.reduce_sum(yt * yp, axis=-1)
    den = tf.norm(yt, axis=-1) * tf.norm(yp, axis=-1) + 1e-8
    return tf.reduce_mean(1.0 - num / den)


def frequency_loss(y_true, y_pred):
    """
    Penalise power-spectrum differences within the physiological BPM band.
    Both inputs are cast to float32 first (needed under mixed precision).
    """
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)

    yt_fft = tf.abs(tf.signal.rfft(yt))
    yp_fft = tf.abs(tf.signal.rfft(yp))

    # Keep only physiological frequencies
    n_freq  = tf.shape(yt_fft)[-1]
    fs      = cfg.FPS
    freqs   = tf.cast(tf.range(n_freq), tf.float32) * fs / tf.cast(
                  cfg.CLIP_LEN, tf.float32)
    bpm_mask = tf.cast(
        (freqs >= cfg.BPM_LOW / 60.0) & (freqs <= cfg.BPM_HIGH / 60.0),
        tf.float32)

    yt_band = yt_fft * bpm_mask
    yp_band = yp_fft * bpm_mask

    yt_norm = yt_band / (tf.reduce_max(yt_band, axis=-1, keepdims=True) + 1e-8)
    yp_norm = yp_band / (tf.reduce_max(yp_band, axis=-1, keepdims=True) + 1e-8)

    return tf.reduce_mean(tf.square(yt_norm - yp_norm))


def combined_loss(y_true, y_pred):
    """
    FIX 9: MSE removed — it conflicts with Pearson (amplitude vs shape).
    Using 60% Pearson + 40% frequency-domain supervision.
    """
    return 0.6 * pearson_loss(y_true, y_pred) + 0.4 * frequency_loss(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7 — Temporal smoothness regulariser (callback, not in loss)
# Applied as a post-hoc penalty logged per epoch for monitoring purposes.
# Injecting it into the graph loss would conflict with mixed precision casting.
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSmoothnessCallback(keras.callbacks.Callback):
    """
    Logs the mean first-difference (temporal roughness) of predictions
    on a small validation batch. A decreasing value means smoother signals.
    """
    def __init__(self, val_ds, log_every: int = 5):
        super().__init__()
        self.val_ds     = val_ds
        self.log_every  = log_every
        self.history_ts = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every != 0:
            return
        model = self.model
        if model is None:
            return
        roughness = []
        for X_batch, _ in self.val_ds.take(4):
            preds = model(X_batch, training=False).numpy()  # (B, T)
            diff  = np.diff(preds, axis=-1)
            roughness.append(np.mean(np.abs(diff)))
        mean_r = float(np.mean(roughness))
        self.history_ts.append(mean_r)
        print(f"\n  [TemporalSmoothness] epoch {epoch+1}: "
              f"mean |Δpred| = {mean_r:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class PearsonCorrelation(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="pearson_r", **kwargs)
        self._sum   = tf.Variable(tf.constant(0.0, dtype=tf.float32),
                                  trainable=False)
        self._count = tf.Variable(tf.constant(0.0, dtype=tf.float32),
                                  trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
        yp = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        r  = tf.reduce_sum(yt * yp, axis=-1) / (
             tf.norm(yt, axis=-1) * tf.norm(yp, axis=-1) + 1e-8)
        self._sum.assign_add(tf.reduce_sum(r))
        self._count.assign_add(tf.cast(tf.size(r), tf.float32))

    def result(self):
        return self._sum / (self._count + 1e-8)

    def reset_state(self):
        self._sum.assign(tf.constant(0.0, dtype=tf.float32))
        self._count.assign(tf.constant(0.0, dtype=tf.float32))


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr      = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        ws    = tf.cast(self.warmup_steps, tf.float32)
        ts    = tf.cast(self.total_steps,  tf.float32)
        warmup = self.peak_lr * step / (ws + 1e-8)
        cosine = self.peak_lr * 0.5 * (
            1.0 + tf.cos(np.pi * (step - ws) / (ts - ws + 1e-8)))
        return tf.where(step < ws, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr,
                "warmup_steps": self.warmup_steps,
                "total_steps":  self.total_steps}


# ─────────────────────────────────────────────────────────────────────────────
# Vital Signs Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_vitals(model: keras.Model, clip: np.ndarray, fs: float = cfg.FPS):
    rppg_raw  = model(clip[np.newaxis], training=False).numpy()[0]
    rppg_filt = bandpass_filter(rppg_raw, fs)
    bpm       = signal_to_bpm(rppg_filt, fs)
    red_mean  = clip[:, :, :, 0].mean(axis=(1, 2))
    spo2      = estimate_spo2(rppg_filt, red_mean)
    return rppg_filt, bpm, spo2


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train():
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print("=" * 60)
    print("  rPPG Training — UBFC-rPPG (PhysNet v2, all fixes applied)")
    print("=" * 60)

    print("\n[1/4] Loading dataset …")
    X, y, subj_ids = build_dataset(cfg.DATA_DIR)

    # FIX 3: subject-level split — no leakage from overlapping clips
    X_train, y_train, X_val, y_val = subject_split(X, y, subj_ids)

    train_ds = make_tf_dataset(X_train, y_train, training=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   training=False)

    print("\n[2/4] Building model …")
    model = build_physnet(cfg.CLIP_LEN, cfg.IMG_H, cfg.IMG_W)
    model.summary()

    param_bytes      = sum(np.prod(v.shape) for v in model.trainable_variables) * 2
    vram_estimate_mb = (param_bytes * 4) / (1024 ** 2)
    print(f"  Estimated VRAM (rough): ~{vram_estimate_mb:.0f} MB / 8192 MB")

    steps_per_epoch = max(1, len(X_train) // cfg.BATCH_SIZE)
    total_steps     = steps_per_epoch * cfg.EPOCHS
    warmup_steps    = steps_per_epoch * 2

    lr_schedule = WarmupCosineDecay(cfg.LR, warmup_steps, total_steps)
    optimizer   = keras.optimizers.AdamW(
        learning_rate=cast(Any, lr_schedule),  # type: ignore[arg-type]
        weight_decay=1e-4,
    )
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[PearsonCorrelation(), keras.metrics.MeanSquaredError(name="mse")]
    )

    print("\n[3/4] Setting up callbacks …")
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_physnet.keras")
    ts_cb     = TemporalSmoothnessCallback(val_ds, log_every=5)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_pearson_r", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_pearson_r", mode="max",
            patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(os.path.join(cfg.LOG_DIR, "training_log.csv")),
        keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR, update_freq="epoch"),
        ts_cb,
    ]

    print("\n[4/4] Training …\n")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg.EPOCHS, callbacks=callbacks, verbose="auto",
    )

    best_r    = max(history.history['val_pearson_r'])
    best_loss = min(history.history['val_loss'])
    print(f"\nBest val Pearson r : {best_r:.4f}")
    print(f"Best val loss      : {best_loss:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray):
    print("\n── Evaluation on validation set ──")
    bpm_errors, spo2_vals, pearson_rs = [], [], []

    for i in range(min(len(X_val), 20)):
        clip   = X_val[i]
        gt_sig = y_val[i]

        rppg_pred, bpm_pred, spo2_pred = predict_vitals(model, clip)
        gt_filt = bandpass_filter(gt_sig, cfg.FPS)
        bpm_gt  = signal_to_bpm(gt_filt, cfg.FPS)

        yt = gt_filt   - np.mean(gt_filt)
        yp = rppg_pred - np.mean(rppg_pred)
        r  = np.dot(yt, yp) / (np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-8)

        pearson_rs.append(r)
        bpm_errors.append(abs(bpm_pred - bpm_gt))
        spo2_vals.append(spo2_pred)

    print(f"  Mean |BPM error| : {np.mean(bpm_errors):.2f} BPM")
    print(f"  Mean Pearson r   : {np.mean(pearson_rs):.4f}")
    print(f"  Mean SpO2 (est.) : {np.mean(spo2_vals):.1f} %")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],         label="train loss")
    axes[0].plot(history.history["val_loss"],     label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["pearson_r"],    label="train r")
    axes[1].plot(history.history["val_pearson_r"],label="val r")
    axes[1].set_title("Pearson r")
    axes[1].legend()
    plt.tight_layout()
    out_path = os.path.join(cfg.LOG_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Training curves saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference Demo
# ─────────────────────────────────────────────────────────────────────────────

def inference_demo(model_path: str, video_path: str):
    print(f"\nLoading model from {model_path} …")
    model = cast(keras.Model, keras.models.load_model(
        model_path,
        custom_objects={
            "combined_loss":      combined_loss,
            "PearsonCorrelation": PearsonCorrelation,
            "WarmupCosineDecay":  WarmupCosineDecay,
        }
    ))

    cap       = cv2.VideoCapture(video_path)
    extractor = FaceROIExtractor()
    frames    = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extractor.extract(frame, cfg.IMG_H, cfg.IMG_W))
    cap.release()
    print(f"  Face miss rate: {extractor.face_miss_rate():.0%}")

    frames    = np.array(frames, dtype=np.float32)
    n         = (len(frames) // cfg.CLIP_LEN) * cfg.CLIP_LEN
    frames    = frames[:n]
    rppg_full = []
    for start in range(0, n, cfg.CLIP_LEN):
        clip = frames[start: start + cfg.CLIP_LEN]
        if len(clip) < cfg.CLIP_LEN:
            break
        rppg_full.append(model(clip[np.newaxis], training=False).numpy()[0])

    rppg_full = np.concatenate(rppg_full)
    rppg_filt = bandpass_filter(rppg_full, cfg.FPS)
    bpm       = signal_to_bpm(rppg_filt, cfg.FPS)
    red_mean  = frames[:len(rppg_filt), :, :, 0].mean(axis=(1, 2))
    spo2      = estimate_spo2(rppg_filt, red_mean)

    print(f"\n  Predicted BPM  : {bpm:.1f}")
    print(f"  Estimated SpO2 : {spo2:.1f} %")

    t = np.arange(len(rppg_filt)) / cfg.FPS
    plt.figure(figsize=(12, 3))
    plt.plot(t, rppg_filt, color="tomato", linewidth=0.8)
    plt.title(f"rPPG Signal — {bpm:.1f} BPM | SpO2 ≈ {spo2:.1f}%")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("rppg_prediction.png", dpi=120)
    plt.close()
    print("  Signal plot → rppg_prediction.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="rPPG Training & Inference")
    parser.add_argument("--mode",  choices=["train", "infer"], default="train")
    parser.add_argument("--model", default="checkpoints/best_physnet.keras")
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        model, history = train()
        X, y, subj_ids = build_dataset(cfg.DATA_DIR)
        _, _, X_val_ev, y_val_ev = subject_split(X, y, subj_ids)
        evaluate(model, X_val_ev, y_val_ev)
        plot_history(history)

    elif args.mode == "infer":
        if args.video is None:
            parser.error("--video is required for --mode infer")
        inference_demo(args.model, args.video)