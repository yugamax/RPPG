import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {dev}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ✅")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return dev

SEQ_LEN     = 256
BATCH_SIZE  = 32
LR          = 1e-4          # 🔥 lower LR to reduce overfitting speed
EPOCHS      = 100
PATIENCE    = 25
NUM_WORKERS = 4

class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None, augment=False):
        self.samples = []
        self.augment = augment
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
        x = x.copy()

        if self.augment:
            # 🔥 Much stronger augmentation
            # 1. Gaussian noise (stronger)
            x += np.random.uniform(0.02, 0.08) * np.random.randn(*x.shape).astype(np.float32)

            # 2. Random channel scaling (simulate skin tone / lighting variation)
            scale = np.random.uniform(0.85, 1.15, size=(3, 1)).astype(np.float32)
            x *= scale

            # 3. Random DC offset per channel
            x += np.random.uniform(-0.1, 0.1, size=(3, 1)).astype(np.float32)

            # 4. Random temporal flip (PPG is quasi-periodic, flip is valid)
            if np.random.rand() < 0.3:
                x = x[:, ::-1].copy()
                y = y[::-1].copy()

            # 5. Random amplitude scaling of the signal target
            amp = np.random.uniform(0.9, 1.1)
            y = (y * amp).astype(np.float32)
            # Re-normalize y after scaling
            y = (y - y.mean()) / (y.std() + 1e-8)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(fps, dtype=torch.float32),
        )


class DilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            ch_in, ch_out, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn      = nn.BatchNorm1d(ch_out)
        self.act     = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)   # 🔥 dropout after each block

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))


class TSCAN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            DilatedBlock(3,  24, dilation=1, dropout=dropout),   # 🔥 smaller
            DilatedBlock(24, 32, dilation=2, dropout=dropout),
            DilatedBlock(32, 32, dilation=4, dropout=dropout),
            DilatedBlock(32, 24, dilation=8, dropout=dropout),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(24, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 24),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(24, 1, kernel_size=1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        attn = self.channel_attn(feat).unsqueeze(-1)
        feat = feat * attn
        return self.head(feat).squeeze(1)


def pearson_loss(pred, target):
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num    = (pred * target).sum(dim=1)
    den    = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return 1 - (num / den).mean()


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
        checkpoint = torch.load("checkpoint.pth", map_location=device, weights_only=True)  # 🔥 fix warning
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        print(f"🔁 Resuming from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"], checkpoint["best_val"], checkpoint["counter"]

    print("🆕 Starting fresh training")
    return 0, float("inf"), 0


def get_or_create_subject_split(processed_dir, split_file="subject_split.json", train_ratio=0.8):
    subjects = sorted(
        set(f.split("_")[0] for f in os.listdir(processed_dir) if f.endswith("_rgb.npy"))
    )

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split_data = json.load(f)
        print(f"Loaded existing subject split from {split_file}")
        return split_data["train_subjects"], split_data["val_subjects"]

    shuffled = subjects.copy()
    random.shuffle(shuffled)
    split_idx = int(train_ratio * len(shuffled))
    split_data = {
        "train_subjects": shuffled[:split_idx],
        "val_subjects":   shuffled[split_idx:],
    }
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"Generated and saved subject split to {split_file}")
    return split_data["train_subjects"], split_data["val_subjects"]


def train():
    global device
    device = setup_device()

    processed_dir = "processed"
    train_subjects, val_subjects = get_or_create_subject_split(processed_dir)

    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects  ({len(val_subjects)}):   {val_subjects}\n")

    train_ds = SubjectDataset(processed_dir, train_subjects, augment=True)   # 🔥 augment=True
    val_ds   = SubjectDataset(processed_dir, val_subjects, stride=SEQ_LEN, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model     = TSCAN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)  # 🔥 stronger decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    start_epoch, best_val, counter = load_checkpoint(model, optimizer, scheduler, scaler)

    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'LR':>10} | Gap")
    print("-" * 56)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                pred = model(x)
                loss = pearson_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(device_type=device.type):
                    pred = model(x)
                    loss = pearson_loss(pred, y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        gap         = val_loss - train_loss

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0
            flag = " ✅"
        else:
            counter += 1
            flag = ""

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {lr:>10.6f} | {gap:+.4f}{flag}")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val, counter)

        if counter >= PATIENCE:
            print(f"\n⛔ Early stopping at epoch {epoch}")
            break

    print(f"\n✅ Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()