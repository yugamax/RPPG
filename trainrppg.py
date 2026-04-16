import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {dev}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ✅")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark   = True
    return dev

# =====================
# CONFIG
# =====================
SEQ_LEN     = 256    # back to 256 - more windows per subject
BATCH_SIZE  = 32     # can go back up
LR          = 5e-5   # 🔥 Stage 2: lower LR for stability
EPOCHS      = 120
PATIENCE    = 40     # give it more time
NUM_WORKERS = 4
GRAD_CLIP   = 1.0    # 🔥 Stage 2: gradient clipping

# 🔥 Stage 1: Green channel emphasis weights (R, G, B)
# Green carries the strongest rPPG signal (~60% of PPG info)
CHANNEL_WEIGHTS = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float32)


# =====================
# DATASET
# =====================
class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN,
                 stride=None, augment=False):
        self.samples  = []
        self.augment  = augment
        stride        = stride or seq_len // 8

        for subj in subjects:
            rgb  = np.load(os.path.join(processed_dir, f"{subj}_rgb.npy"))
            ppg  = np.load(os.path.join(processed_dir, f"{subj}_ppg.npy"))
            fps  = np.load(os.path.join(processed_dir, f"{subj}_fps.npy"))[0]

            # 🔥 Stage 4: Load POS/CHROM if available, else zeros
            pos_path   = os.path.join(processed_dir, f"{subj}_pos.npy")
            chrom_path = os.path.join(processed_dir, f"{subj}_chrom.npy")
            pos   = np.load(pos_path)   if os.path.exists(pos_path)   else np.zeros(len(rgb), dtype=np.float32)
            chrom = np.load(chrom_path) if os.path.exists(chrom_path) else np.zeros(len(rgb), dtype=np.float32)

            for start in range(0, len(rgb) - seq_len, stride):
                clip      = rgb[start:start + seq_len].T.astype(np.float32)   # (3, T)
                signal    = ppg[start:start + seq_len].astype(np.float32)
                pos_clip  = pos[start:start + seq_len].astype(np.float32)
                chrom_clip= chrom[start:start + seq_len].astype(np.float32)
                self.samples.append((clip, signal, fps, pos_clip, chrom_clip))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fps, pos, chrom = self.samples[idx]
        x     = x.copy()
        y     = y.copy()

        if self.augment:
            # 🔥 Stage 1: Strong augmentation

            # 1. Gaussian noise (channel-aware: more on R/B, less on G)
            noise_scale        = np.random.uniform(0.01, 0.04)
            channel_noise      = np.array([1.2, 0.8, 1.2])[:, None]  # protect green
            x += (noise_scale * channel_noise * np.random.randn(*x.shape)).astype(np.float32)

            # 2. Random per-channel scaling (simulate skin tone / lighting)
            scale = np.random.uniform(0.90, 1.10, size=(3, 1)).astype(np.float32)
            x *= scale

            # 3. Random DC offset (lighting drift)
            x += np.random.uniform(-0.08, 0.08, size=(3, 1)).astype(np.float32)

            # 4. Temporal flip (PPG is periodic, flip is valid)
            if np.random.rand() < 0.3:
                x     = x[:, ::-1].copy()
                y     = y[::-1].copy()
                pos   = pos[::-1].copy()
                chrom = chrom[::-1].copy()

            # 5. Random amplitude jitter on target
            amp = np.random.uniform(0.85, 1.15)
            y   = (y * amp).astype(np.float32)
            y   = ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)

            # 6. Random time shift (circular roll)
            shift = np.random.randint(0, x.shape[1])
            x     = np.roll(x,     shift, axis=1)
            y     = np.roll(y,     shift)
            pos   = np.roll(pos,   shift)
            chrom = np.roll(chrom, shift)

            # 7. Frequency masking on RGB (like SpecAugment)
            if np.random.rand() < 0.3:
                mask_start  = np.random.randint(0, x.shape[1] - 10)
                mask_len    = np.random.randint(5, 20)
                x[:, mask_start:mask_start + mask_len] = 0.0

        # 🔥 Stage 1: Stack 5 input channels: RGB + POS + CHROM
        pos_t   = pos[np.newaxis, :]    # (1, T)
        chrom_t = chrom[np.newaxis, :]  # (1, T)
        x_full  = np.concatenate([x, pos_t, chrom_t], axis=0)  # (5, T)

        return (
            torch.from_numpy(x_full),
            torch.from_numpy(y),
            torch.tensor(fps, dtype=torch.float32),
        )


# =====================
# MODEL
# =====================

# 🔥 Stage 3: Temporal Attention module
class TemporalAttention(nn.Module):
    """
    Lightweight self-attention along time axis.
    Queries/keys are computed over a downsampled sequence
    to keep memory manageable for long sequences.
    """
    def __init__(self, channels, num_heads=4, downsample=4):
        super().__init__()
        self.downsample = downsample
        self.attn       = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.norm       = nn.LayerNorm(channels)
        self.pool       = nn.AvgPool1d(downsample, stride=downsample)
        self.upsample   = nn.Upsample(scale_factor=downsample, mode='nearest')

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        # Downsample for efficiency
        x_down = self.pool(x)                      # (B, C, T//ds)
        x_perm = x_down.permute(0, 2, 1)           # (B, T//ds, C)
        x_norm = self.norm(x_perm)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # self-attention
        attn_out    = attn_out.permute(0, 2, 1)           # (B, C, T//ds)

        # Upsample back and add residual
        attn_up = self.upsample(attn_out)          # (B, C, T) approx
        if attn_up.shape[-1] != T:
            attn_up = F.interpolate(attn_up, size=T, mode='nearest')

        return x + attn_up


class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation=1, dropout=0.15):
        super().__init__()
        self.conv    = nn.Conv1d(
            ch_in, ch_out, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn      = nn.BatchNorm1d(ch_out)
        self.act     = nn.GELU()                   # GELU > ReLU for sequence models
        self.dropout = nn.Dropout(dropout)

        # Residual projection if channels differ
        self.proj    = nn.Conv1d(ch_in, ch_out, 1, bias=False) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x)))) + self.proj(x)


# 🔥 Stage 3: Simplified architecture with temporal attention
class TSCAN(nn.Module):
    """
    Input: (B, 5, T) — 3 RGB channels + POS + CHROM
    Output: (B, T)
    """
    def __init__(self, in_channels=5, dropout=0.15):
        super().__init__()

        # 🔥 Stage 1: Green channel emphasis via learned input weighting
        self.channel_weight = nn.Parameter(
            torch.tensor([0.2, 0.6, 0.2, 0.5, 0.5], dtype=torch.float32)
        )

        # 🔥 Stage 3: Simplified encoder
        self.encoder = nn.Sequential(
            DilatedBlock(in_channels, 32, dilation=1,  dropout=dropout),
            DilatedBlock(32,          32, dilation=2,  dropout=dropout),
            DilatedBlock(32,          48, dilation=4,  dropout=dropout),
            DilatedBlock(48,          48, dilation=8,  dropout=dropout),
            DilatedBlock(48,          32, dilation=16, dropout=dropout),
        )

        # 🔥 Stage 3: Temporal attention after encoding
        self.temporal_attn = TemporalAttention(channels=32, num_heads=4, downsample=8)

        # Channel attention (squeeze-excitation)
        self.channel_attn  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.GELU(),
            nn.Linear(8, 32),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        # 🔥 Stage 1: Soft channel weighting (green emphasis)
        w    = torch.sigmoid(self.channel_weight).view(1, -1, 1)
        x    = x * w

        feat = self.encoder(x)

        # 🔥 Stage 3: Temporal self-attention
        feat = self.temporal_attn(feat)

        # 🔥 NEW: dropout after attention (reduces overfitting)
        feat = F.dropout(feat, p=0.2, training=self.training)

        # Channel attention
        attn = self.channel_attn(feat).unsqueeze(-1)
        feat = feat * attn

        return self.head(feat).squeeze(1)   # (B, T)


# =====================
# LOSS FUNCTIONS
# =====================
def pearson_loss(pred, target):
    """1 - Pearson correlation: 0 = perfect, 2 = worst."""
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num    = (pred * target).sum(dim=1)
    den    = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return 1 - (num / den).mean()


# 🔥 Stage 1: Frequency-domain loss
def frequency_loss(pred, target, fps=30.0, low_hz=0.7, high_hz=4.0):
    """
    Compare power spectral density of pred vs target in the
    physiologically relevant band [0.7, 4.0] Hz.
    Encourages the model to learn the correct rPPG frequency, not just shape.
    """
    B, T    = pred.shape
    freq    = torch.fft.rfftfreq(T, d=1.0 / fps).to(pred.device)   # (T//2+1,)

    # Bandpass mask
    mask    = (freq >= low_hz) & (freq <= high_hz)

    pred_f  = torch.fft.rfft(pred.float(),   norm="ortho")   # (B, T//2+1)
    tgt_f   = torch.fft.rfft(target.float(), norm="ortho")

    pred_p  = pred_f.abs()[..., mask]                # (B, F_band)
    tgt_p   = tgt_f.abs()[..., mask]

    # Normalize each spectrum so we compare shape, not amplitude
    pred_p  = pred_p  / (pred_p.sum(dim=1, keepdim=True)  + 1e-8)
    tgt_p   = tgt_p   / (tgt_p.sum(dim=1, keepdim=True)   + 1e-8)

    return F.mse_loss(pred_p, tgt_p)


def combined_loss(pred, target, fps_batch, freq_weight=0.3):
    """
    Stage 1: Pearson loss + frequency-domain loss.
    freq_weight controls balance (0 = pure Pearson, 1 = pure frequency).
    """
    l_pearson = pearson_loss(pred, target)

    # Use mean FPS across batch for FFT (close enough)
    fps_val   = fps_batch.float().mean().item()
    l_freq    = frequency_loss(pred, target, fps=fps_val)

    return l_pearson + freq_weight * l_freq, l_pearson.item(), l_freq.item()


# =====================
# CHECKPOINT UTILS
# =====================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val, counter):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_val":  best_val,
        "counter":   counter,
    }, "checkpoint.pth")


def load_checkpoint(model, optimizer, scheduler, scaler):
    if os.path.exists("checkpoint.pth"):
        try:
            ckpt = torch.load("checkpoint.pth", map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            print(f"🔁 Resuming from epoch {ckpt['epoch']}")
            return ckpt["epoch"], ckpt["best_val"], ckpt["counter"]
        except Exception as e:
            print(f"⚠️  Could not load checkpoint ({e}). Starting fresh.")

    print("🆕 Starting fresh training")
    return 0, float("inf"), 0


# =====================
# SUBJECT SPLIT
# =====================
def get_or_create_subject_split(processed_dir, split_file="subject_split.json", train_ratio=0.8):
    subjects = sorted(
        set(f.split("_")[0] for f in os.listdir(processed_dir) if f.endswith("_rgb.npy"))
    )
    print(f"Total processed subjects found: {len(subjects)}")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split_data = json.load(f)
        # Validate saved split against what's actually on disk
        missing_train = [s for s in split_data["train_subjects"] if s not in subjects]
        missing_val   = [s for s in split_data["val_subjects"]   if s not in subjects]
        if missing_train or missing_val:
            print(f"⚠️  Split has subjects not in processed dir: {missing_train + missing_val}")
            print("   Regenerating split...")
        else:
            print(f"Loaded existing split from {split_file}")
            return split_data["train_subjects"], split_data["val_subjects"]

    shuffled  = subjects.copy()
    random.shuffle(shuffled)
    split_idx = int(train_ratio * len(shuffled))
    split_data = {
        "train_subjects": shuffled[:split_idx],
        "val_subjects":   shuffled[split_idx:],
    }
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"Generated and saved split to {split_file}")
    return split_data["train_subjects"], split_data["val_subjects"]


# =====================
# TRAIN
# =====================
def train():
    global device
    device = setup_device()

    processed_dir = "processed"
    train_subjects, val_subjects = get_or_create_subject_split(processed_dir)

    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects  ({len(val_subjects)}):   {val_subjects}\n")

    train_ds = SubjectDataset(processed_dir, train_subjects, augment=True)
    val_ds   = SubjectDataset(processed_dir, val_subjects, stride=SEQ_LEN // 2, augment=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model     = TSCAN(in_channels=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2   # warm restarts help escape local minima
    )
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    start_epoch, best_val, counter = load_checkpoint(model, optimizer, scheduler, scaler)

    print(f"{'Ep':>4} | {'Train':>8} | {'Val':>8} | {'Pearson':>8} | {'Freq':>8} | {'LR':>9} | Gap")
    print("-" * 72)

    for epoch in range(start_epoch + 1, EPOCHS + 1):

        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        train_pearson = 0.0
        train_freq    = 0.0

        for x, y, fps in train_loader:
            x, y, fps = x.to(device), y.to(device), fps.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                pred              = model(x)
                loss, l_p, l_f   = combined_loss(pred, y, fps)

            scaler.scale(loss).backward()

            # 🔥 Stage 2: Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item()
            train_pearson += l_p
            train_freq    += l_f

        # ── VALIDATE ──
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, fps in val_loader:
                x, y, fps = x.to(device), y.to(device), fps.to(device)
                with autocast(device_type=device.type):
                    pred       = model(x)
                    loss, _, _ = combined_loss(pred, y, fps)
                val_loss += loss.item()

        n_train = len(train_loader)
        n_val   = len(val_loader)

        train_loss    /= n_train
        train_pearson /= n_train
        train_freq    /= n_train
        val_loss      /= n_val
        gap            = val_loss - train_loss

        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            counter  = 0
            flag     = " ✅"
        else:
            counter += 1
            flag     = ""

        print(
            f"{epoch:>4} | {train_loss:>8.4f} | {val_loss:>8.4f} | "
            f"{train_pearson:>8.4f} | {train_freq:>8.4f} | "
            f"{lr:>9.6f} | {gap:+.4f}{flag}"
        )

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val, counter)

        if counter >= PATIENCE:
            print(f"\n⛔ Early stopping at epoch {epoch}")
            break

    print(f"\n✅ Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()