import os
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# PATHS
# =====================
DATA_ROOT = "UBFC_DATASET/DATASET_2"
SAVE_DIR  = "processed"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# MEDIAPIPE FACE MESH
# =====================
# Each worker thread gets its own FaceMesh instance to avoid race conditions
def make_face_mesh():
    face_mesh_module = getattr(mp.solutions, "face_mesh")
    return face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# Landmark indices for three ROIs on the face
FOREHEAD   = [10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK = [50, 101, 118, 119, 120, 121]
RIGHT_CHEEK= [280, 330, 347, 348, 349, 350]

def fill_roi(mask, landmark_obj, indices, h, w):
    """Fill a convex-hull ROI on the mask from a set of landmark indices."""
    pts = []
    for i in indices:
        p = landmark_obj.landmark[i]
        pts.append([int(p.x * w), int(p.y * h)])
    pts = np.array(pts, np.int32)
    hull = cv2.convexHull(pts)           # fix: convexHull before fillConvexPoly
    cv2.fillConvexPoly(mask, hull, 255)

def get_roi_mean(frame_rgb, face_mesh_instance):
    """
    Returns mean RGB values over forehead + cheek ROIs.
    No manual channel boosting — let the model learn weighting.
    Returns None if no face detected.
    """
    h, w, _ = frame_rgb.shape
    result = face_mesh_instance.process(frame_rgb)
    if not result.multi_face_landmarks:
        return None

    lm   = result.multi_face_landmarks[0]
    mask = np.zeros((h, w), dtype=np.uint8)

    fill_roi(mask, lm, FOREHEAD,    h, w)
    fill_roi(mask, lm, LEFT_CHEEK,  h, w)
    fill_roi(mask, lm, RIGHT_CHEEK, h, w)

    pixels = frame_rgb[mask == 255]      # shape (N, 3) — RGB
    if len(pixels) == 0:
        return None

    return pixels.mean(axis=0)           # mean R, G, B — no artificial boost

# =====================
# SIGNAL PROCESSING
# =====================
def bandpass(sig, fs, lo=0.7, hi=4.0):
    nyq = fs / 2.0
    if nyq <= 0:
        raise ValueError("Invalid Nyquist frequency")

    lo_adj = max(lo, 0.01)
    hi_adj = min(hi, nyq * 0.95)
    if lo_adj >= hi_adj:
        raise ValueError(f"Invalid bandpass range for fs={fs}: lo={lo_adj}, hi={hi_adj}")

    ba = butter(3, [lo_adj / nyq, hi_adj / nyq], btype='band', output='ba')
    if not isinstance(ba, tuple) or len(ba) != 2:
        raise ValueError("Unexpected filter coefficient format")
    b, a = ba
    return filtfilt(b, a, sig)

def normalize(sig):
    sig = detrend(sig)
    return (sig - sig.mean()) / (sig.std() + 1e-8)

def normalize_rgb(rgb_signal):
    """Per-channel normalize so the model sees zero-mean unit-variance input."""
    out = rgb_signal.copy().astype(np.float32)
    for c in range(3):
        out[:, c] = normalize(out[:, c])
    return out

def parse_ground_truth(gt):
    """Extract PPG and timestamp arrays from UBFC ground-truth text."""
    arr = np.asarray(gt, dtype=np.float64)

    if arr.ndim == 1:
        raise ValueError("ground_truth.txt has 1D shape; expected at least 3 series")

    if arr.shape[0] >= 3:
        ppg = arr[0]
        t = arr[2]
    elif arr.shape[1] >= 3:
        ppg = arr[:, 0]
        t = arr[:, 2]
    else:
        raise ValueError(f"Unsupported ground-truth shape: {arr.shape}")

    ppg = np.asarray(ppg, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()

    valid = np.isfinite(ppg) & np.isfinite(t)
    ppg = ppg[valid]
    t = t[valid]
    if len(ppg) < 2 or len(t) < 2:
        raise ValueError("Not enough valid GT samples")

    order = np.argsort(t)
    t = t[order]
    ppg = ppg[order]

    if np.allclose(t[0], t[-1]):
        raise ValueError("GT timestamps are constant")

    return ppg, t

# =====================
# PROCESS ONE SUBJECT
# =====================
def process_subject(subj):
    subj_path  = os.path.join(DATA_ROOT, subj)
    video_path = os.path.join(subj_path, "vid.avi")
    gt_path    = os.path.join(subj_path, "ground_truth.txt")

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        return f"[SKIP] {subj} — missing files"

    # Load ground truth
    try:
        gt = np.loadtxt(gt_path)
        ppg, t = parse_ground_truth(gt)
    except Exception as e:
        return f"[SKIP] {subj} — bad ground truth ({e})"

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return f"[SKIP] {subj} — bad FPS"

    # Each thread has its own FaceMesh
    face_mesh_instance = make_face_mesh()

    rgb_signal  = []
    timestamps  = []
    bad_frames  = 0
    idx         = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_rgb  = get_roi_mean(frame_rgb, face_mesh_instance)

            if mean_rgb is None:
                bad_frames += 1
                # Fill with previous value or zeros to maintain temporal alignment
                mean_rgb = rgb_signal[-1] if rgb_signal else np.zeros(3)

            rgb_signal.append(mean_rgb)
            timestamps.append(idx / fps)
            idx += 1
    finally:
        cap.release()
        face_mesh_instance.close()

    if len(rgb_signal) < 64:
        return f"[SKIP] {subj} — too short ({len(rgb_signal)} frames)"

    rgb_signal = np.array(rgb_signal, dtype=np.float32)   # (T, 3)
    timestamps = np.array(timestamps, dtype=np.float64)

    # Align PPG to video timestamps
    aligned_ppg = np.interp(timestamps, t, ppg)
    try:
        aligned_ppg = bandpass(aligned_ppg, fps)
    except ValueError as e:
        return f"[SKIP] {subj} — filtering failed ({e})"
    aligned_ppg = normalize(aligned_ppg).astype(np.float32)

    # Normalize RGB per channel BEFORE saving
    rgb_signal = normalize_rgb(rgb_signal)

    # Save
    np.save(os.path.join(SAVE_DIR, f"{subj}_rgb.npy"), rgb_signal)
    np.save(os.path.join(SAVE_DIR, f"{subj}_ppg.npy"), aligned_ppg)
    np.save(os.path.join(SAVE_DIR, f"{subj}_fps.npy"), np.array([fps], dtype=np.float32))

    return f"[OK] {subj} — {idx} frames, {bad_frames} bad ({100*bad_frames/max(idx,1):.1f}%)"

# =====================
# MAIN
# =====================
def preprocess():
    if not os.path.isdir(DATA_ROOT):
        print(f"Dataset path not found: {DATA_ROOT}")
        return

    subjects = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    print(f"Found {len(subjects)} subjects: {subjects}\n")

    if not subjects:
        print("No subject folders found. Nothing to preprocess.")
        return

    # Parallel processing — MediaPipe CPU is the bottleneck here,
    # so threading across subjects speeds things up significantly.
    # Keep workers ≤ CPU cores; 4 is safe for most systems.
    max_workers = min(4, len(subjects))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_subject, s): s for s in subjects}
        for fut in as_completed(futures):
            subj = futures[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f"[ERR] {subj} — unexpected failure ({e})")

    print("\n✅ Preprocessing complete")

if __name__ == "__main__":
    preprocess()