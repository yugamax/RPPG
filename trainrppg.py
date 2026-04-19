import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# =====================
# SEED
# =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# =====================
# CONFIG
# =====================
SEQ_LEN     = 256
BATCH_SIZE  = 8
ACCUM_STEPS = 4        # effective batch = 32
LR          = 2e-4
EPOCHS      = 150
PATIENCE    = 20
GRAD_CLIP   = 1.0

# =====================
# FREQ MASK CACHE
# =====================
_freq_mask_cache = {}

def get_freq_mask(T, fps_val, dev):
    key = (T, round(fps_val, 2))
    if key not in _freq_mask_cache:
        freq = torch.fft.rfftfreq(T, d=1.0 / fps_val)
        _freq_mask_cache[key] = ((freq >= 0.7) & (freq <= 4.0)).to(dev)
    return _freq_mask_cache[key]

# =====================
# DATASET
# 5 input channels: R, G, B, POS, CHROM
# =====================
class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None, augment=False):
        self.samples = []
        self.augment = augment
        stride = stride or seq_len

        for subj in subjects:
            rgb_path   = os.path.join(processed_dir, f"{subj}_rgb.npy")
            ppg_path   = os.path.join(processed_dir, f"{subj}_ppg.npy")
            fps_path   = os.path.join(processed_dir, f"{subj}_fps.npy")
            pos_path   = os.path.join(processed_dir, f"{subj}_pos.npy")
            chrom_path = os.path.join(processed_dir, f"{subj}_chrom.npy")

            if not all(os.path.exists(p) for p in [rgb_path, ppg_path, fps_path]):
                continue

            rgb  = np.load(rgb_path)    # (T, 3)
            ppg  = np.load(ppg_path)    # (T,)
            fps  = np.load(fps_path)[0]

            # POS and CHROM — load if available, else zeros
            pos   = np.load(pos_path).reshape(-1, 1)   if os.path.exists(pos_path)   else np.zeros((len(rgb), 1), np.float32)
            chrom = np.load(chrom_path).reshape(-1, 1) if os.path.exists(chrom_path) else np.zeros((len(rgb), 1), np.float32)

            # Stack into (T, 5): R G B POS CHROM
            combined = np.concatenate([rgb, pos, chrom], axis=1).astype(np.float32)

            min_len  = min(len(combined), len(ppg))
            combined = combined[:min_len]
            ppg      = ppg[:min_len]

            for start in range(0, min_len - seq_len, stride):
                clip   = combined[start:start+seq_len].T   # (5, T)
                signal = ppg[start:start+seq_len]
                self.samples.append((clip, signal, float(fps)))

        print(f"  Loaded {len(self.samples)} samples from {len(subjects)} subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fps = self.samples[idx]
        x = x.copy().astype(np.float32)
        y = y.copy().astype(np.float32)

        # Normalize each channel independently
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        if self.augment:
            # Noise + scale + shift only on RGB (channels 0-2)
            # Physics signals (3-4) are left clean — corrupting them hurts more than it helps
            x[:3] += np.random.normal(0, 0.05, x[:3].shape).astype(np.float32)
            x[:3] *= np.random.uniform(0.9, 1.1, size=(3, 1)).astype(np.float32)
            x[:3] += np.random.uniform(-0.1, 0.1, size=(3, 1)).astype(np.float32)

            # Temporal flip — applied to ALL channels consistently
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
                y = y[::-1].copy()

            # Low-freq drift on RGB only
            t     = np.linspace(0, 2 * np.pi, x.shape[1]).astype(np.float32)
            drift = (np.random.uniform(-0.08, 0.08) * np.sin(np.random.uniform(0.01, 0.1) * t))
            x[:3] += drift[np.newaxis, :]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(fps, dtype=torch.float32)
        )

# =====================
# MODEL
# Split encoder: RGB and physics signals processed separately
# then merged — stops physics signals dominating RGB stream
# =====================
class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, 3, padding=dilation, dilation=dilation)
        self.bn   = nn.BatchNorm1d(ch_out)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x)))) + self.skip(x)


class TSCAN(nn.Module):
    def __init__(self):
        super().__init__()

        # RGB stream: 3 → 32
        self.rgb_stem = nn.Sequential(
            DilatedBlock(3,  32, 1, dropout=0.0),
            DilatedBlock(32, 32, 2, dropout=0.0),
        )

        # Physics stream (POS + CHROM): 2 → 32
        self.phys_stem = nn.Sequential(
            DilatedBlock(2,  32, 1, dropout=0.0),
            DilatedBlock(32, 32, 2, dropout=0.0),
        )

        # Shared encoder after merge: 64 → 64
        self.encoder = nn.Sequential(
            DilatedBlock(64, 96, 4,  dropout=0.1),
            DilatedBlock(96, 96, 8,  dropout=0.1),
            DilatedBlock(96, 64, 16, dropout=0.15),
        )

        self.attn      = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.norm      = nn.LayerNorm(64)
        self.attn_drop = nn.Dropout(0.1)
        self.head      = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        rgb_feat  = self.rgb_stem(x[:, :3, :])   # (B, 32, T)
        phys_feat = self.phys_stem(x[:, 3:, :])  # (B, 32, T)

        feat = torch.cat([rgb_feat, phys_feat], dim=1)  # (B, 64, T)
        feat = self.encoder(feat)

        f = feat.permute(0, 2, 1)
        attn_out, _ = self.attn(f, f, f)
        f = self.attn_drop(self.norm(f + attn_out))
        f = f.permute(0, 2, 1)

        return self.head(f).squeeze(1)

# =====================
# LOSSES
# =====================
def pearson_loss(pred, target):
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return 1 - (num / den).mean()


def frequency_loss(pred, target, fps_val):
    B, T = pred.shape
    mask   = get_freq_mask(T, fps_val, pred.device)
    pred_f = torch.fft.rfft(pred.float())
    tgt_f  = torch.fft.rfft(target.float())
    pred_p = pred_f.abs()[..., mask]
    tgt_p  = tgt_f.abs()[..., mask]
    pred_p = pred_p / (pred_p.sum(dim=1, keepdim=True) + 1e-8)
    tgt_p  = tgt_p  / (tgt_p.sum(dim=1, keepdim=True) + 1e-8)
    return F.mse_loss(pred_p, tgt_p)


def combined_loss(pred, target, fps):
    pred   = pred.float()
    target = target.float()
    l1 = pearson_loss(pred, target)
    l2 = frequency_loss(pred, target, fps.mean().item())
    return 0.7 * l1 + 1.5 * l2

# =====================
# TRAIN
# =====================
def train():
    processed_dir = "processed"

    subjects = sorted(set(
        f.split("_")[0] for f in os.listdir(processed_dir)
        if f.endswith("_rgb.npy")
    ))
    print(f"Found {len(subjects)} subjects: {subjects}\n")

    # Stratified split — every 5th subject goes to val
    val_subj   = subjects[::5]
    train_subj = [s for s in subjects if s not in set(val_subj)]
    print(f"Train: {len(train_subj)} subjects | Val: {len(val_subj)} subjects")

    print("\nLoading train set...")
    train_ds = SubjectDataset(processed_dir, train_subj, stride=SEQ_LEN // 2, augment=True)
    print("Loading val set...")
    val_ds   = SubjectDataset(processed_dir, val_subj,   stride=SEQ_LEN,       augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=2, pin_memory=True)

    model  = TSCAN().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    best     = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()

        # --- Train ---
        model.train()
        train_loss = 0.0
        opt.zero_grad()

        for i, (x, y, fps) in enumerate(train_loader):
            x, y, fps = x.to(device), y.to(device), fps.to(device)

            with autocast(device_type=device.type):
                pred = model(x)
                loss = combined_loss(pred, y, fps) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS

        # --- Val ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, fps in val_loader:
                x, y, fps = x.to(device), y.to(device), fps.to(device)
                with autocast(device_type=device.type):
                    pred = model(x)
                    val_loss += combined_loss(pred, y, fps).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        sched.step()

        gap = val_loss - train_loss
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}  Val={val_loss:.4f}  Gap={gap:+.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best val: {best:.4f}")
            break

    print(f"\nTraining complete. Best val loss: {best:.4f}")


if __name__ == "__main__":
    train()