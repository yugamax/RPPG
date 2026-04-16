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
SEQ_LEN     = 256
BATCH_SIZE  = 32
LR          = 3e-4
EPOCHS      = 120
PATIENCE    = 30
GRAD_CLIP   = 1.0

# =====================
# DATASET
# =====================
class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None, augment=False):
        self.samples = []
        self.augment = augment
        stride = stride or seq_len  # 🔥 no overlap

        for subj in subjects:
            rgb = np.load(os.path.join(processed_dir, f"{subj}_rgb.npy"))
            ppg = np.load(os.path.join(processed_dir, f"{subj}_ppg.npy"))
            fps = np.load(os.path.join(processed_dir, f"{subj}_fps.npy"))[0]

            for start in range(0, len(rgb) - seq_len, stride):
                clip   = rgb[start:start+seq_len].T.astype(np.float32)
                signal = ppg[start:start+seq_len].astype(np.float32)
                self.samples.append((clip, signal, fps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fps = self.samples[idx]

        # Normalize
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        if self.augment:
            noise = np.random.normal(0, 0.05, x.shape).astype(np.float32)
            x += noise

            scale = np.random.uniform(0.9, 1.1, size=(3,1)).astype(np.float32)
            x *= scale

        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(fps, dtype=torch.float32)

# =====================
# MODEL
# =====================
class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, 3, padding=dilation, dilation=dilation)
        self.bn   = nn.BatchNorm1d(ch_out)
        self.act  = nn.GELU()
        self.skip = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + self.skip(x)


class TSCAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            DilatedBlock(3, 64, 1),
            DilatedBlock(64, 64, 2),
            DilatedBlock(64, 96, 4),
            DilatedBlock(96, 96, 8),
            DilatedBlock(96, 64, 16),
        )

        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        feat = self.encoder(x)

        feat = feat.permute(0,2,1)
        feat, _ = self.attn(feat, feat, feat)
        feat = feat.permute(0,2,1)

        return self.head(feat).squeeze(1)

# =====================
# LOSSES
# =====================
def pearson_loss(pred, target):
    pred   = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return 1 - (num / den).mean()


def frequency_loss(pred, target, fps):
    B, T = pred.shape
    freq = torch.fft.rfftfreq(T, d=1.0/fps).to(pred.device)
    mask = (freq >= 0.7) & (freq <= 4.0)

    pred_f = torch.fft.rfft(pred.float())
    tgt_f  = torch.fft.rfft(target.float())

    pred_p = pred_f.abs()[..., mask]
    tgt_p  = tgt_f.abs()[..., mask]

    pred_p = pred_p / (pred_p.sum(dim=1, keepdim=True)+1e-8)
    tgt_p  = tgt_p  / (tgt_p.sum(dim=1, keepdim=True)+1e-8)

    return F.mse_loss(pred_p, tgt_p)


def combined_loss(pred, target, fps):
    l1 = pearson_loss(pred, target)
    l2 = frequency_loss(pred, target, fps.mean().item())
    return 0.7*l1 + 1.5*l2

# =====================
# TRAIN
# =====================
def train():
    processed_dir = "processed"

    subjects = sorted(set(f.split("_")[0] for f in os.listdir(processed_dir)))
    random.shuffle(subjects)

    split = int(0.8 * len(subjects))
    train_subj = subjects[:split]
    val_subj   = subjects[split:]

    train_ds = SubjectDataset(processed_dir, train_subj, stride=SEQ_LEN//2, augment=True)
    val_ds   = SubjectDataset(processed_dir, val_subj, stride=SEQ_LEN, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TSCAN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler(enabled=(device.type=="cuda"))

    best = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x,y,fps in train_loader:
            x,y,fps = x.to(device), y.to(device), fps.to(device)

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

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x,y,fps in val_loader:
                x,y,fps = x.to(device), y.to(device), fps.to(device)
                pred = model(x)
                val_loss += combined_loss(pred, y, fps).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        sched.step()

        print(f"Epoch {epoch+1}: Train={train_loss:.4f} Val={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping")
            break


if __name__ == "__main__":
    train()