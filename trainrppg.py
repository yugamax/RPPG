import os
import glob
import numpy as np
import tensorflow as tf
import keras
from keras import layers
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
    CLIP_LEN        = 128          # frames per clip
    IMG_H           = 128          # spatial resize
    IMG_W           = 128
    BATCH_SIZE      = 4
    EPOCHS          = 50
    LR              = 1e-4
    VAL_SPLIT       = 0.2
    SEED            = 42
    FPS             = 30.0         # UBFC camera FPS
    BPM_LOW         = 40           # physiological BPM range
    BPM_HIGH        = 240
    CHECKPOINT_DIR  = "checkpoints"
    LOG_DIR         = "logs"
    FACE_CASCADE: str | None = None
    GT_FILENAME     = "gtdump.xmp"


def _face_cascade_path() -> str:
    return os.path.join(getattr(cv2, "data").haarcascades, "haarcascade_frontalface_default.xml")


cfg = Config()
cfg.FACE_CASCADE = _face_cascade_path()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)

def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0) -> np.ndarray:
    nyq = 0.5 * fs
    sos = sp_signal.butter(3, [low / nyq, high / nyq], btype="band", output="sos")
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


def estimate_spo2(signal: np.ndarray) -> float:
    ac = np.std(signal)
    dc = np.abs(np.mean(signal)) + 1e-8
    r  = ac / dc
    spo2 = 110.0 - 25.0 * r
    return float(np.clip(spo2, 70.0, 100.0))


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    mu, sd = np.mean(sig), np.std(sig) + 1e-8
    return (sig - mu) / sd

def parse_ground_truth(gt_path: str):
    data = np.loadtxt(gt_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ppg         = data[:, 3] if data.shape[1] > 3 else np.zeros(len(data))
    return ppg.astype(np.float32)

class FaceROIExtractor:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cfg.FACE_CASCADE or "")
        self._last_roi = None

    def extract(self, frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            self._last_roi = (x, y, w, h)

        if self._last_roi is not None:
            x, y, w, h = self._last_roi
            roi = frame[y: y + h, x: x + w]
        else:
            roi = frame

        return cv2.resize(roi, (out_w, out_h)).astype(np.float32) / 255.0

def load_video_clips(video_path: str, gt_ppg: np.ndarray, clip_len: int):
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

    clips, targets = [], []
    for start in range(0, n_frames - clip_len + 1, clip_len // 2):
        end = start + clip_len
        clip_ppg = normalize_signal(gt_ppg[start:end])
        clips.append(frames[start:end])
        targets.append(clip_ppg)

    if not clips:
        return None, None

    return np.array(clips, dtype=np.float32), np.array(targets, dtype=np.float32)

def build_dataset(data_dir: str):
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
        clips, targets = load_video_clips(vid_path, gt_ppg, cfg.CLIP_LEN)

        if clips is None:
            print("no clips extracted")
            continue

        all_clips.append(clips)
        all_targets.append(targets)
        print(f"{len(clips)} clips")

    X = np.concatenate(all_clips,   axis=0)
    y = np.concatenate(all_targets, axis=0)
    print(f"\nTotal: {X.shape[0]} clips | X {X.shape} | y {y.shape}")
    return X, y

def make_tf_dataset(X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=cfg.SEED)
    ds = ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def conv_bn_relu(x, filters, kernel, padding="same", strides=(1, 1, 1)):
    x = layers.Conv3D(filters, kernel, strides=strides, padding=padding,
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_physnet(clip_len: int, img_h: int, img_w: int) -> keras.Model:
    inp = keras.Input(shape=(clip_len, img_h, img_w, 3), name="video_clip")

    x = conv_bn_relu(inp, 32, (1, 5, 5))
    x = conv_bn_relu(x,  32, (3, 3, 3))
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = conv_bn_relu(x,  64, (3, 3, 3))
    x = conv_bn_relu(x,  64, (3, 3, 3))
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = conv_bn_relu(x, 128, (3, 3, 3))
    x = conv_bn_relu(x, 128, (3, 3, 3))
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = conv_bn_relu(x, 256, (3, 3, 3))
    x = conv_bn_relu(x, 256, (3, 3, 3))
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=[2, 3]),
                       name="spatial_avg")(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(64,  3, padding="same", activation="relu")(x)
    x = layers.Conv1D(1,   1, padding="same")(x)
    out = layers.Lambda(lambda t: tf.squeeze(t, axis=-1),
                         name="rppg_signal")(x)

    return keras.Model(inp, out, name="PhysNet")

def pearson_loss(y_true, y_pred):
    yt = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    yp = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    num  = tf.reduce_sum(yt * yp, axis=-1)
    den  = (tf.norm(yt, axis=-1) * tf.norm(yp, axis=-1)) + 1e-8
    r    = num / den
    return tf.reduce_mean(1.0 - r)


def combined_loss(y_true, y_pred):
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    corr = pearson_loss(y_true, y_pred)
    return 0.5 * mse + 0.5 * corr

class PearsonCorrelation(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="pearson_r", **kwargs)
        self._sum: tf.Variable = tf.Variable(tf.constant(0.0), trainable=False, dtype=tf.float32)
        self._count: tf.Variable = tf.Variable(tf.constant(0.0), trainable=False, dtype=tf.float32)

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

def predict_vitals(model: keras.Model, clip: np.ndarray, fs: float = cfg.FPS):
    clip_input = clip[np.newaxis, ...]
    rppg_raw   = model(clip_input, training=False).numpy()[0]
    rppg_filt  = bandpass_filter(rppg_raw, fs)
    bpm        = signal_to_bpm(rppg_filt, fs)
    spo2       = estimate_spo2(rppg_filt)
    return rppg_filt, bpm, spo2

def train():
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print("=" * 60)
    print("  rPPG Training — UBFC-rPPG Dataset")
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

    optimizer = keras.optimizers.Adam(learning_rate=cfg.LR)
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[PearsonCorrelation(), keras.metrics.MeanSquaredError(name="mse")]
    )

    print("\n[3/4] Setting up callbacks …")
    ckpt_path  = os.path.join(cfg.CHECKPOINT_DIR, "best_physnet.keras")
    callbacks  = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_pearson_r", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_pearson_r", mode="max",
            patience=12, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR, histogram_freq=0),
        keras.callbacks.CSVLogger(os.path.join(cfg.LOG_DIR, "training_log.csv")),
    ]

    print("\n[4/4] Training …\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
        verbose="auto",
    )

    return model, history

def evaluate(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray):
    print("\n── Evaluation on validation set ──")
    bpm_errors, spo2_vals, pearson_rs = [], [], []

    for i in range(min(len(X_val), 20)):
        clip   = X_val[i]
        gt_sig = y_val[i]

        rppg_pred, bpm_pred, spo2_pred = predict_vitals(model, clip)
        gt_filt  = bandpass_filter(gt_sig, cfg.FPS)
        bpm_gt   = signal_to_bpm(gt_filt, cfg.FPS)

        yt = gt_filt  - np.mean(gt_filt)
        yp = rppg_pred - np.mean(rppg_pred)
        r  = np.dot(yt, yp) / (np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-8)
        pearson_rs.append(r)
        bpm_errors.append(abs(bpm_pred - bpm_gt))
        spo2_vals.append(spo2_pred)

    print(f"  Mean |BPM error| : {np.mean(bpm_errors):.2f} BPM")
    print(f"  Mean Pearson r   : {np.mean(pearson_rs):.4f}")
    print(f"  Mean SpO2 (est.) : {np.mean(spo2_vals):.1f} %")

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

def inference_demo(model_path: str, video_path: str):
    print(f"\nLoading model from {model_path} …")
    model = cast(keras.Model, keras.models.load_model(
        model_path,
        custom_objects={"combined_loss": combined_loss,
                        "PearsonCorrelation": PearsonCorrelation}
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
    spo2      = estimate_spo2(rppg_filt)

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

        # X, y = build_dataset(cfg.DATA_DIR)
        # _, X_val, _, y_val = train_test_split(
        #     X, y, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED
        # )
        # evaluate(model, X_val, y_val)
        # plot_history(history)

    elif args.mode == "infer":
        if args.video is None:
            parser.error("--video is required for --mode infer")
        inference_demo(args.model, args.video)