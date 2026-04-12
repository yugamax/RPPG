import os
import glob
import numpy as np

# Disable MKL and oneDNN to avoid primitive creation errors on Windows
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Configure GPU
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
from numpy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from typing import cast
warnings.filterwarnings("ignore")


class Config:
    DATA_DIR        = "data/UBFC_DATASET/DATASET_1"
    CLIP_LEN        = 96          # balanced: more context than 64, less VRAM than 128
    IMG_H           = 64          # keep spatial modest to save VRAM
    IMG_W           = 64
    BATCH_SIZE      = 2           # safe for 8GB with residual+attention model
    EPOCHS          = 40
    LR              = 3e-4
    VAL_SPLIT       = 0.2
    SEED            = 42
    FPS             = 30.0
    BPM_LOW         = 40
    BPM_HIGH        = 240
    CHECKPOINT_DIR  = "checkpoints"
    LOG_DIR         = "logs"
    FACE_CASCADE: str | None = None
    GT_FILENAME     = "gtdump.xmp"
    STRIDE          = 48          # stride relative to clip_len=96
    AUGMENT         = True
    MIXED_PRECISION = True        # fp16 cuts VRAM usage ~40% with negligible accuracy loss


def _face_cascade_path() -> str:
    return os.path.join(getattr(cv2, "data").haarcascades, "haarcascade_frontalface_default.xml")


cfg = Config()
cfg.FACE_CASCADE = _face_cascade_path()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)

# Mixed precision: fp16 compute, fp32 weights → ~40% less VRAM on RTX 4060
if cfg.MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled (fp16)")


# ─────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0) -> np.ndarray:
    nyq = 0.5 * fs
    sos = sp_signal.butter(4, [low / nyq, high / nyq], btype="band", output="sos")  # ↑ order 3→4
    return sp_signal.sosfiltfilt(sos, signal)


def signal_to_bpm(signal: np.ndarray, fs: float) -> float:
    n = len(signal)
    freqs = rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(rfft(signal - np.mean(signal)))
    mask = (freqs >= cfg.BPM_LOW / 60.0) & (freqs <= cfg.BPM_HIGH / 60.0)
    if not np.any(mask):
        return 0.0
    peak_freq = freqs[mask][np.argmax(spectrum[mask])]
    return peak_freq * 60.0


def estimate_spo2(red_signal: np.ndarray, ir_signal: np.ndarray | None = None) -> float:
    """
    Improved SpO2 estimation.
    If only one channel available, falls back to single-channel approximation.
    """
    if ir_signal is not None and len(ir_signal) > 0:
        ac_red = np.std(red_signal)
        dc_red = np.abs(np.mean(red_signal)) + 1e-8
        ac_ir  = np.std(ir_signal)
        dc_ir  = np.abs(np.mean(ir_signal)) + 1e-8
        r = (ac_red / dc_red) / (ac_ir / dc_ir)
    else:
        ac = np.std(red_signal)
        dc = np.abs(np.mean(red_signal)) + 1e-8
        r  = ac / dc

    # Standard empirical formula
    spo2 = 110.0 - 25.0 * r
    return float(np.clip(spo2, 70.0, 100.0))


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    mu, sd = np.mean(sig), np.std(sig) + 1e-8
    return (sig - mu) / sd


# ─────────────────────────────────────────────
# Ground Truth Parsing  (UBFC-rPPG format)
# ─────────────────────────────────────────────

def parse_ground_truth(gt_path: str) -> np.ndarray:
    """
    UBFC gtdump.xmp stores rows as: [time, HR, SpO2, ppg_value, ...]
    We try multiple column indices and pick the one with highest variance
    (most likely to be the raw PPG waveform).
    """
    try:
        data = np.loadtxt(gt_path, delimiter=',')
    except Exception:
        # Some files use space delimiter
        data = np.loadtxt(gt_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 1:
        return data[:, 0].astype(np.float32)

    # Pick column with highest normalized variance — likely raw PPG
    variances = [np.std(data[:, c]) / (np.abs(np.mean(data[:, c])) + 1e-8)
                 for c in range(data.shape[1])]
    best_col = int(np.argmax(variances))
    ppg = data[:, best_col]
    return ppg.astype(np.float32)


# ─────────────────────────────────────────────
# Face ROI Extractor
# ─────────────────────────────────────────────

class FaceROIExtractor:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cfg.FACE_CASCADE or "")
        self._last_roi = None
        self._no_face_count = 0

    def extract(self, frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            # Expand ROI slightly to include forehead (rich in rPPG signal)
            pad_y = int(h * 0.1)
            y = max(0, y - pad_y)
            h = min(frame.shape[0] - y, h + pad_y)
            self._last_roi = (x, y, w, h)
            self._no_face_count = 0
        else:
            self._no_face_count += 1

        if self._last_roi is not None:
            x, y, w, h = self._last_roi
            roi = frame[y: y + h, x: x + w]
        else:
            roi = frame

        # Convert BGR→RGB for better channel semantics
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return cv2.resize(roi_rgb, (out_w, out_h)).astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────

def augment_clip(clip: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a (T, H, W, 3) clip."""
    # Random horizontal flip
    if np.random.rand() < 0.5:
        clip = clip[:, :, ::-1, :]

    # Random brightness/contrast jitter
    alpha = np.random.uniform(0.85, 1.15)   # contrast
    beta  = np.random.uniform(-0.05, 0.05)  # brightness
    clip  = np.clip(clip * alpha + beta, 0.0, 1.0)

    # Random temporal reversal (rPPG signal is roughly symmetric)
    if np.random.rand() < 0.3:
        clip = clip[::-1]

    return clip.astype(np.float32)


# ─────────────────────────────────────────────
# Video Loading
# ─────────────────────────────────────────────

def load_video_clips(video_path: str, gt_ppg: np.ndarray, clip_len: int, stride: int):
    cap = cv2.VideoCapture(video_path)
    extractor = FaceROIExtractor()
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = extractor.extract(frame, cfg.IMG_H, cfg.IMG_W)
        frames.append(roi)
    cap.release()

    frames = np.array(frames, dtype=np.float32)
    n_frames = min(len(frames), len(gt_ppg))
    frames   = frames[:n_frames]
    gt_ppg   = gt_ppg[:n_frames]

    # Pre-filter ground truth PPG
    if n_frames > 10:
        gt_ppg = bandpass_filter(gt_ppg, cfg.FPS)

    clips, targets = [], []
    # Sliding window with stride for more training samples
    for start in range(0, n_frames - clip_len + 1, stride):
        end      = start + clip_len
        clip_ppg = normalize_signal(gt_ppg[start:end])
        clips.append(frames[start:end])
        targets.append(clip_ppg)

    if not clips:
        return None, None

    return np.array(clips, dtype=np.float32), np.array(targets, dtype=np.float32)


# ─────────────────────────────────────────────
# Dataset I/O
# ─────────────────────────────────────────────

def save_dataset(X: np.ndarray, y: np.ndarray, data_dir: str):
    np.savez(os.path.join(data_dir, "dataset.npz"), X=X, y=y)
    print("Dataset saved to dataset.npz")


def load_dataset(data_dir: str):
    path = os.path.join(data_dir, "dataset.npz")
    if os.path.exists(path):
        data = np.load(path)
        return data['X'], data['y']
    return None, None


def build_dataset(data_dir: str):
    X, y = load_dataset(data_dir)
    if X is not None and y is not None:
        print("Loaded dataset from cache")
        return X, y

    subject_dirs = sorted(
        path for path in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(path)
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No subjects found under {data_dir}")

    all_clips, all_targets = [], []

    for subj in subject_dirs:
        vid_path = os.path.join(subj, "vid.avi")
        gt_path  = os.path.join(subj, cfg.GT_FILENAME)

        if not (os.path.exists(vid_path) and os.path.exists(gt_path)):
            print(f"  [SKIP] {subj} — missing files")
            continue

        print(f"  Loading {os.path.basename(subj)} …", end=" ", flush=True)
        gt_ppg = parse_ground_truth(gt_path)
        clips, targets = load_video_clips(vid_path, gt_ppg, cfg.CLIP_LEN, cfg.STRIDE)

        if clips is None:
            print("no clips extracted")
            continue

        all_clips.append(clips)
        all_targets.append(targets)
        print(f"{len(clips)} clips")

    X = np.concatenate(all_clips,   axis=0)
    y = np.concatenate(all_targets, axis=0)
    print(f"\nTotal: {X.shape[0]} clips | X {X.shape} | y {y.shape}")
    save_dataset(X, y, data_dir)
    return X, y


# ─────────────────────────────────────────────
# tf.data pipeline with augmentation
# ─────────────────────────────────────────────

def augment_tf(clip, target):
    clip = tf.numpy_function(
        lambda c: augment_clip(c.numpy()), [clip], tf.float32
    )
    clip.set_shape([cfg.CLIP_LEN, cfg.IMG_H, cfg.IMG_W, 3])
    return clip, target


def make_tf_dataset(X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=cfg.SEED)
        if cfg.AUGMENT:
            ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────
# Model — Improved PhysNet with Attention
# ─────────────────────────────────────────────

def conv_bn_relu(x, filters, kernel, padding="same", strides=(1, 1, 1)):
    x = layers.Conv3D(filters, kernel, strides=strides, padding=padding,
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def residual_3d_block(x, filters):
    """3D residual block for better gradient flow."""
    shortcut = x
    x = conv_bn_relu(x, filters, (3, 3, 3))
    x = layers.Conv3D(filters, (3, 3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Match channels if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, (1, 1, 1), padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def temporal_attention(x):
    """Lightweight self-attention — capped head dim to save VRAM."""
    # x shape: (B, T, C)
    C = x.shape[-1]
    head_dim = min(C // 4, 32)   # cap at 32 to limit memory on 8GB GPU

    q = layers.Dense(head_dim, use_bias=False)(x)
    k = layers.Dense(head_dim, use_bias=False)(x)
    v = layers.Dense(C,        use_bias=False)(x)

    scores  = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(head_dim, tf.float32))
    weights = tf.nn.softmax(scores, axis=-1)
    out     = tf.matmul(weights, v)

    # Residual
    x = layers.Add()([x, out])
    x = layers.LayerNormalization()(x)
    return x


def build_physnet(clip_len: int, img_h: int, img_w: int) -> keras.Model:
    inp = keras.Input(shape=(clip_len, img_h, img_w, 3), name="video_clip")

    # ── Stem ──────────────────────────────────
    x = conv_bn_relu(inp, 32, (1, 5, 5))
    x = conv_bn_relu(x,  32, (3, 3, 3))
    x = layers.MaxPool3D((1, 2, 2))(x)          # spatial ↓

    # ── Stage 1 ───────────────────────────────
    x = residual_3d_block(x, 64)
    x = layers.MaxPool3D((1, 2, 2))(x)

    # ── Stage 2 ───────────────────────────────
    x = residual_3d_block(x, 128)
    x = layers.MaxPool3D((1, 2, 2))(x)

    # ── Stage 3 ───────────────────────────────
    x = residual_3d_block(x, 256)
    x = layers.MaxPool3D((2, 2, 2))(x)          # temporal ↓ too

    # ── Stage 4 ───────────────────────────────
    x = residual_3d_block(x, 256)
    x = layers.MaxPool3D((1, 2, 2))(x)

    # ── Spatial pooling → temporal sequence ───
    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=[2, 3]),
                      name="spatial_avg")(x)     # (B, T', C)

    # ── Temporal attention ────────────────────
    x = layers.Lambda(lambda t: temporal_attention(t),
                      name="temporal_attn")(x)

    # ── Temporal decoder ──────────────────────
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64,  3, padding="same", activation="relu")(x)

    # Upsample back to clip_len if temporal dim was halved
    x = layers.Lambda(lambda t: tf.image.resize(
        tf.expand_dims(t, -1), [clip_len, 1], method="bilinear"
    ))(x)
    x = layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(x)   # (B, clip_len, 64)

    x = layers.Conv1D(1, 1, padding="same")(x)
    out = layers.Lambda(lambda t: tf.squeeze(t, axis=-1),
                        name="rppg_signal")(x)   # (B, clip_len)

    return keras.Model(inp, out, name="PhysNet_v2")


# ─────────────────────────────────────────────
# Losses & Metrics
# ─────────────────────────────────────────────

def pearson_loss(y_true, y_pred):
    yt  = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    yp  = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    num = tf.reduce_sum(yt * yp, axis=-1)
    den = (tf.norm(yt, axis=-1) * tf.norm(yp, axis=-1)) + 1e-8
    r   = num / den
    return tf.reduce_mean(1.0 - r)


def frequency_loss(y_true, y_pred):
    """Penalize frequency-domain differences — key for BPM accuracy."""
    # Cast to complex for FFT
    yt_fft = tf.abs(tf.signal.rfft(y_true))
    yp_fft = tf.abs(tf.signal.rfft(y_pred))
    # Normalize
    yt_fft = yt_fft / (tf.reduce_max(yt_fft, axis=-1, keepdims=True) + 1e-8)
    yp_fft = yp_fft / (tf.reduce_max(yp_fft, axis=-1, keepdims=True) + 1e-8)
    return tf.reduce_mean(tf.square(yt_fft - yp_fft))


def combined_loss(y_true, y_pred):
    mse   = tf.reduce_mean(tf.square(y_true - y_pred))
    corr  = pearson_loss(y_true, y_pred)
    freq  = frequency_loss(y_true, y_pred)
    # Weighted combination
    return 0.3 * mse + 0.5 * corr + 0.2 * freq


class PearsonCorrelation(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="pearson_r", **kwargs)
        self._sum: tf.Variable   = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._count: tf.Variable = tf.Variable(0.0, trainable=False, dtype=tf.float32)

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
        self._sum.assign(0.0)
        self._count.assign(0.0)


# ─────────────────────────────────────────────
# Learning Rate Schedule: Warmup + Cosine Decay
# ─────────────────────────────────────────────

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr      = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        ws    = tf.cast(self.warmup_steps, tf.float32)
        ts    = tf.cast(self.total_steps,  tf.float32)
        warmup_lr = self.peak_lr * (step / (ws + 1e-8))
        cosine_lr = self.peak_lr * 0.5 * (
            1.0 + tf.cos(np.pi * (step - ws) / (ts - ws + 1e-8))
        )
        return tf.where(step < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr":      self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps":  self.total_steps,
        }


# ─────────────────────────────────────────────
# Vital Signs Prediction
# ─────────────────────────────────────────────

def predict_vitals(model: keras.Model, clip: np.ndarray, fs: float = cfg.FPS):
    clip_input = clip[np.newaxis, ...]
    rppg_raw   = model(clip_input, training=False).numpy()[0]
    rppg_filt  = bandpass_filter(rppg_raw, fs)
    bpm        = signal_to_bpm(rppg_filt, fs)
    # Use red channel mean as proxy for SpO2 estimation
    red_mean   = clip[:, :, :, 0].mean(axis=(1, 2))
    spo2       = estimate_spo2(rppg_filt, red_mean)
    return rppg_filt, bpm, spo2


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train():
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print("=" * 60)
    print("  rPPG Training — UBFC-rPPG Dataset (PhysNet v2)")
    print("=" * 60)

    print("\n[1/4] Loading dataset …")
    X, y = build_dataset(cfg.DATA_DIR)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED, shuffle=True
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    train_ds = make_tf_dataset(X_train, y_train, training=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   training=False)

    print("\n[2/4] Building model …")
    model = build_physnet(cfg.CLIP_LEN, cfg.IMG_H, cfg.IMG_W)
    model.summary()

    # Estimate VRAM: warn if likely near limit
    param_bytes = sum(np.prod(v.shape) for v in model.trainable_variables) * 2  # fp16
    vram_estimate_mb = (param_bytes * 4) / (1024 ** 2)  # rough: 4x params for gradients/optimizer
    print(f"  Estimated VRAM (rough): ~{vram_estimate_mb:.0f} MB — RTX 4060 has 8192 MB")

    # Warmup for first 5% of steps, then cosine decay
    steps_per_epoch = max(1, len(X_train) // cfg.BATCH_SIZE)
    total_steps     = steps_per_epoch * cfg.EPOCHS
    warmup_steps    = steps_per_epoch * 2   # 2 warmup epochs

    lr_schedule = WarmupCosineDecay(cfg.LR, warmup_steps, total_steps)
    optimizer   = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[PearsonCorrelation(), keras.metrics.MeanSquaredError(name="mse")]
    )

    print("\n[3/4] Setting up callbacks …")
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_physnet.keras")
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
        keras.callbacks.TensorBoard(
            log_dir=cfg.LOG_DIR, histogram_freq=0, update_freq="epoch"
        ),
    ]

    print("\n[4/4] Training …\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
        verbose="auto",
    )

    final_val_pearson = max(history.history['val_pearson_r'])
    final_val_loss    = min(history.history['val_loss'])
    print(f"\nBest Validation Pearson Correlation: {final_val_pearson:.4f}")
    print(f"Best Validation Loss: {final_val_loss:.4f}")

    return model, history


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray):
    print("\n── Evaluation on validation set ──")
    bpm_errors, spo2_vals, pearson_rs = [], [], []

    n_eval = min(len(X_val), 20)
    for i in range(n_eval):
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


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["pearson_r"],     label="train r")
    axes[1].plot(history.history["val_pearson_r"], label="val r")
    axes[1].set_title("Pearson r")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(cfg.LOG_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nTraining curves saved → {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# Inference Demo
# ─────────────────────────────────────────────

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

    print(f"Reading video {video_path} …")
    cap       = cv2.VideoCapture(video_path)
    extractor = FaceROIExtractor()
    frames    = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extractor.extract(frame, cfg.IMG_H, cfg.IMG_W))
    cap.release()

    frames = np.array(frames, dtype=np.float32)
    n      = (len(frames) // cfg.CLIP_LEN) * cfg.CLIP_LEN
    frames = frames[:n]

    rppg_full = []
    for start in range(0, n, cfg.CLIP_LEN):
        clip = frames[start: start + cfg.CLIP_LEN]
        if len(clip) < cfg.CLIP_LEN:
            break
        pred = model(clip[np.newaxis], training=False).numpy()[0]
        rppg_full.append(pred)

    rppg_full = np.concatenate(rppg_full)
    rppg_filt = bandpass_filter(rppg_full, cfg.FPS)
    bpm       = signal_to_bpm(rppg_filt, cfg.FPS)
    # Use red channel mean from all frames as proxy
    red_mean  = frames[:, :, :, 0].mean(axis=(1, 2))[:len(rppg_filt)]
    spo2      = estimate_spo2(rppg_filt, red_mean)

    print(f"\n  Predicted BPM  : {bpm:.1f}")
    print(f"  Estimated SpO2 : {spo2:.1f} %")

    plt.figure(figsize=(12, 3))
    t = np.arange(len(rppg_filt)) / cfg.FPS
    plt.plot(t, rppg_filt, color="tomato", linewidth=0.8)
    plt.title(f"Predicted rPPG Signal — {bpm:.1f} BPM | SpO2 ≈ {spo2:.1f}%")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("rppg_prediction.png", dpi=120)
    plt.close()
    print("  Signal plot saved → rppg_prediction.png")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="rPPG Training & Inference")
    parser.add_argument("--mode",  choices=["train", "infer"], default="train")
    parser.add_argument("--model", default="checkpoints/best_physnet.keras",
                        help="Path to saved .keras model (for --mode infer)")
    parser.add_argument("--video", default=None,
                        help="Path to video file (for --mode infer)")
    args = parser.parse_args()

    if args.mode == "train":
        model, history = train()
        X, y = build_dataset(cfg.DATA_DIR)
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED
        )
        evaluate(model, X_val, y_val)
        plot_history(history)

    elif args.mode == "infer":
        if args.video is None:
            parser.error("--video is required for --mode infer")
        inference_demo(args.model, args.video)