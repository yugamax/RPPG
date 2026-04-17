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
print("Using:", device)

# =====================
# CONFIG
# =====================
SEQ_LEN    = 256
BATCH_SIZE = 16      # smaller → noisier gradients → implicit regularization
LR         = 1e-4    # lower LR to slow convergence and stay in the flat region longer
EPOCHS     = 200
PATIENCE   = 20      # tighter patience since val plateaus fast
GRAD_CLIP  = 0.5     # was 1.0 — tighter clip reduces big gradient spikes

# =====================
# DATASET
# =====================
class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None, augment=False):
        self.samples = []
        self.augment = augment
        # FIXED: no overlap between train windows; val uses full stride too
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

        # Normalize per channel
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        if self.augment:
            # Gaussian noise
            x += np.random.normal(0, 0.03, x.shape).astype(np.float32)

            # Per-channel scale jitter
            scale = np.random.uniform(0.85, 1.15, size=(3, 1)).astype(np.float32)
            x *= scale

            # Random temporal flip (HR signal is symmetric)
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
                y = y[::-1].copy()

            # Mixup-style baseline drift (low-freq additive noise)
            t = np.linspace(0, 2 * np.pi, x.shape[1]).astype(np.float32)
            freq = np.random.uniform(0.01, 0.1)
            drift = (np.random.uniform(-0.1, 0.1) * np.sin(freq * t)).astype(np.float32)
            x += drift[np.newaxis, :]

        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(fps, dtype=torch.float32)


# =====================
# MODEL  (slimmer + regularized)
# =====================
class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation, dropout=0.1):
        super().__init__()
        self.conv  = nn.Conv1d(ch_in, ch_out, 3, padding=dilation, dilation=dilation)
        self.bn    = nn.BatchNorm1d(ch_out)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x)))) + self.skip(x)


class TSCAN(nn.Module):
    """
    Slimmer encoder (fewer channels) + attention dropout + output dropout.
    Fewer parameters = less capacity to memorize training set.
    """
    def __init__(self, dropout=0.15):
        super().__init__()

        # Reduced channel widths: 64→48, 96→64
        self.encoder = nn.Sequential(
            DilatedBlock(3,  48, 1,  dropout),
            DilatedBlock(48, 48, 2,  dropout),
            DilatedBlock(48, 64, 4,  dropout),
            DilatedBlock(64, 64, 8,  dropout),
            DilatedBlock(64, 48, 16, dropout),
        )

        # Attention with dropout
        self.attn = nn.MultiheadAttention(48, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(48)

        self.out_drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(48, 1, 1)

    def forward(self, x):
        feat = self.encoder(x)                       # (B, 48, T)

        f = feat.permute(0, 2, 1)                   # (B, T, 48)
        attn_out, _ = self.attn(f, f, f)
        f = self.norm(f + attn_out)                 # pre-norm residual
        f = f.permute(0, 2, 1)                      # (B, 48, T)

        return self.head(self.out_drop(f)).squeeze(1)


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
    # Reduced freq weight: was 1.5 — FFT loss variance was destabilizing val
    return 0.7 * l1 + 0.8 * l2


# =====================
# TRAIN
# =====================
def train():
    processed_dir = "processed"

    subjects = sorted(set(f.split("_")[0] for f in os.listdir(processed_dir)))
    random.shuffle(subjects)

    split      = int(0.8 * len(subjects))
    train_subj = subjects[:split]
    val_subj   = subjects[split:]

    # No overlap in training stride (was SEQ_LEN//2 which inflated dataset artificially)
    train_ds = SubjectDataset(processed_dir, train_subj, stride=SEQ_LEN, augment=True)
    val_ds   = SubjectDataset(processed_dir, val_subj,   stride=SEQ_LEN, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True)

    model = TSCAN().to(device)

    # Weight decay for L2 regularization — separated from bias/norm params
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if "bn" in name or "norm" in name or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-3},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR
    )

    # Warmup + cosine: ramps up for 10 epochs then cosines down
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    patience  = 0

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
        sched.step()

        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}  Val={val_loss:.4f}  LR={current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best val: {best_val:.4f}")
            break


if __name__ == "__main__":
    train()