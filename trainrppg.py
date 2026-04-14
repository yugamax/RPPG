import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ✅")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# =====================
# CONFIG
# =====================
SEQ_LEN     = 256
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 50
PATIENCE    = 7
NUM_WORKERS = 4

# =====================
# DATASET
# =====================
class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None):
        self.samples = []
        stride = stride or seq_len // 2

        for subj in subjects:
            rgb = np.load(os.path.join(processed_dir, f"{subj}_rgb.npy"))
            ppg = np.load(os.path.join(processed_dir, f"{subj}_ppg.npy"))
            fps = np.load(os.path.join(processed_dir, f"{subj}_fps.npy"))[0]

            for start in range(0, len(rgb) - seq_len, stride):
                clip   = rgb[start:start + seq_len].T.astype(np.float32)
                signal = ppg[start:start + seq_len].astype(np.float32)
                self.samples.append((clip, signal, fps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fps = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(fps, dtype=torch.float32),
        )

# =====================
# MODEL
# =====================
class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            ch_in, ch_out, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn  = nn.BatchNorm1d(ch_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TSCAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            DilatedBlock(3,  32, dilation=1),
            DilatedBlock(32, 64, dilation=2),
            DilatedBlock(64, 64, dilation=4),
            DilatedBlock(64, 32, dilation=8),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 32),
            nn.Sigmoid(),
        )

        self.head = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)
        attn = self.channel_attn(feat).unsqueeze(-1)
        feat = feat * attn
        return self.head(feat).squeeze(1)

# =====================
# LOSS
# =====================
def pearson_loss(pred, target):
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num    = (pred * target).sum(dim=1)
    den    = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return 1 - (num / den).mean()

# =====================
# CHECKPOINT UTILS
# =====================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val, counter):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
        "counter": counter,
    }, "checkpoint.pth")


def load_checkpoint(model, optimizer, scheduler, scaler):
    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth", map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        print(f"🔁 Resuming from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"], checkpoint["best_val"], checkpoint["counter"]

    print("🆕 Starting fresh training")
    return 0, float("inf"), 0

# =====================
# TRAIN
# =====================
def train():
    processed_dir = "processed"

    subjects = sorted(
        set(f.split("_")[0] for f in os.listdir(processed_dir) if f.endswith("_rgb.npy"))
    )
    random.shuffle(subjects)

    split          = int(0.8 * len(subjects))
    train_subjects = subjects[:split]
    val_subjects   = subjects[split:]

    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects  ({len(val_subjects)}):   {val_subjects}\n")

    train_ds = SubjectDataset(processed_dir, train_subjects, stride=SEQ_LEN // 2)
    val_ds   = SubjectDataset(processed_dir, val_subjects,   stride=SEQ_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model     = TSCAN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler    = GradScaler(device="cuda")

    # 🔥 Resume support
    start_epoch, best_val, counter = load_checkpoint(model, optimizer, scheduler, scaler)

    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'LR':>10}")
    print("-" * 46)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        # ── TRAIN ──
        model.train()
        train_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                pred = model(x)
                loss = pearson_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # ── VALIDATION ──
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with autocast(device_type="cuda"):
                    pred = model(x)
                    loss = pearson_loss(pred, y)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        current_lr  = optimizer.param_groups[0]["lr"]

        scheduler.step()

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val,
            }, "best_model.pth")
            flag    = " ✅"
            counter = 0
        else:
            counter += 1

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {current_lr:>10.6f}{flag}")

        # 🔥 SAVE CHECKPOINT EVERY EPOCH
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val, counter)
        print("💾 Checkpoint saved")

        if counter >= PATIENCE:
            print(f"\n⛔ Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\n✅ Done. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    train()