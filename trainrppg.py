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
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# CONFIG
# =====================
SEQ_LEN    = 256
BATCH_SIZE = 16
LR         = 2e-4    # raised back up — model was learning too slowly
EPOCHS     = 200
PATIENCE   = 25
GRAD_CLIP  = 0.5

# =====================
# DATASET
# =====================
class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None, augment=False):
        self.samples = []
        self.augment = augment
        self.seq_len = seq_len
        stride = stride or seq_len

        for subj in subjects:
            rgb = np.load(os.path.join(processed_dir, f"{subj}_rgb.npy"))
            ppg = np.load(os.path.join(processed_dir, f"{subj}_ppg.npy"))
            fps = np.load(os.path.join(processed_dir, f"{subj}_fps.npy"))[0]

            for start in range(0, len(rgb) - seq_len, stride):
                clip   = rgb[start:start+seq_len].T.astype(np.float32)
                signal = ppg[start:start+seq_len].astype(np.float32)
                self.samples.append((clip, signal, float(fps)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fps = self.samples[idx]

        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        if self.augment:
            # Gaussian noise — mild
            x += np.random.normal(0, 0.03, x.shape).astype(np.float32)

            # Per-channel scale jitter
            scale = np.random.uniform(0.9, 1.1, size=(3, 1)).astype(np.float32)
            x *= scale

            # Temporal flip
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
                y = y[::-1].copy()

            # Low-freq baseline drift
            t    = np.linspace(0, 2 * np.pi, x.shape[1]).astype(np.float32)
            freq = np.random.uniform(0.01, 0.1)
            drift = (np.random.uniform(-0.08, 0.08) * np.sin(freq * t)).astype(np.float32)
            x += drift[np.newaxis, :]

            # Per-channel mean shift to simulate camera/skin tone variation
            shift = np.random.uniform(-0.15, 0.15, size=(3, 1)).astype(np.float32)
            x += shift

            # Per-channel gamma perturbation to simulate exposure changes
            gamma = np.random.uniform(0.85, 1.15, size=(3, 1)).astype(np.float32)
            x = np.sign(x) * (np.abs(x) ** gamma)

            # Random temporal crop then resize back to fixed length
            if random.random() < 0.4:
                seq_len = self.seq_len
                crop_len = int(seq_len * random.uniform(0.85, 1.0))
                start = random.randint(0, seq_len - crop_len)
                x_crop = x[:, start:start + crop_len]
                y_crop = y[start:start + crop_len]

                x = F.interpolate(
                    torch.from_numpy(x_crop[None]),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                )[0].numpy()
                y = np.interp(
                    np.linspace(0, 1, seq_len),
                    np.linspace(0, 1, crop_len),
                    y_crop,
                ).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(fps, dtype=torch.float32)


# =====================
# MODEL — balanced capacity
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
    """
    Restored to medium capacity (56/72 channels) — between the original (64/96)
    and the over-regularized version (48/64). Dropout only on the deeper layers
    where overfitting risk is highest, not on shallow feature extractors.
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            DilatedBlock(3,  56, 1,  dropout=0.0),   # early layers: no dropout
            DilatedBlock(56, 56, 2,  dropout=0.05),
            DilatedBlock(56, 72, 4,  dropout=0.15),
            DilatedBlock(72, 72, 8,  dropout=0.15),
            DilatedBlock(72, 56, 16, dropout=0.2),
        )

        # Attention: stronger dropout to regularize the most expressive block
        self.attn = nn.MultiheadAttention(56, 4, dropout=0.15, batch_first=True)
        self.norm = nn.LayerNorm(56)
        self.norm_drop = nn.Dropout(0.1)

        self.head = nn.Conv1d(56, 1, 1)

    def forward(self, x):
        feat = self.encoder(x)

        f = feat.permute(0, 2, 1)
        attn_out, _ = self.attn(f, f, f)
        f = self.norm_drop(self.norm(f + attn_out))
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


def frequency_loss(pred, target, fps):
    B, T = pred.shape
    freq = torch.fft.rfftfreq(T, d=1.0 / fps).to(pred.device)
    mask = (freq >= 0.7) & (freq <= 4.0)

    pred_f = torch.fft.rfft(pred.float())
    tgt_f  = torch.fft.rfft(target.float())

    pred_p = pred_f.abs()[..., mask]
    tgt_p  = tgt_f.abs()[..., mask]

    pred_p = pred_p / (pred_p.sum(dim=1, keepdim=True) + 1e-8)
    tgt_p  = tgt_p  / (tgt_p.sum(dim=1, keepdim=True) + 1e-8)

    return F.mse_loss(pred_p, tgt_p)


def combined_loss(pred, target, fps):
    l1 = pearson_loss(pred, target)
    l2 = frequency_loss(pred, target, fps.mean().item())
    return 0.6 * l1 + 0.4 * l2


# =====================
# TRAIN
# =====================
def train():
    processed_dir = "processed"
    print(f"Using: {device}")

    subjects = sorted(set(f.split("_")[0] for f in os.listdir(processed_dir)))
    subjects_sorted = sorted(subjects)
    val_subj = subjects_sorted[::5]
    val_set = set(val_subj)
    train_subj = [s for s in subjects_sorted if s not in val_set]

    train_ds = SubjectDataset(processed_dir, train_subj, stride=SEQ_LEN, augment=True)
    val_ds   = SubjectDataset(processed_dir, val_subj,   stride=SEQ_LEN // 2, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True)

    model = TSCAN().to(device)

    # Weight decay only on weight matrices, not biases or norm layers
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if "bn" in name or "norm" in name or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": 3e-4},   # reduced from 1e-3
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR
    )

    # Shorter warmup (5 epochs) then cosine — dataset is small, warmup of 10 was too slow
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, 80 - warmup)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))

    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    patience  = 0
    val_history = []

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for x, y, fps in train_loader:
            x, y, fps = x.to(device), y.to(device), fps.to(device)
            opt.zero_grad()

            with autocast(device_type=device.type):
                pred = model(x)
                loss = combined_loss(pred, y, fps)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()

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
        val_history.append(val_loss)
        smoothed_val = float(np.mean(val_history[-3:]))
        sched.step()

        gap = val_loss - train_loss
        current_lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}: Train={train_loss:.4f}  Val={val_loss:.4f}  "
            f"SmoothedVal={smoothed_val:.4f}  Gap={gap:+.4f}  LR={current_lr:.2e}"
        )

        if smoothed_val < best_val:
            best_val = smoothed_val
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best val: {best_val:.4f}")
            break


if __name__ == "__main__":
    train()