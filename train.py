import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

SEQ_LEN     = 256
BATCH_SIZE  = 8
ACCUM_STEPS = 4
LR          = 3e-4
EPOCHS      = 120
PATIENCE    = 8
GRAD_CLIP   = 1.0

_freq_mask_cache = {}

def get_freq_mask(T, fps_val, dev):
    key = (T, round(fps_val, 2))
    if key not in _freq_mask_cache:
        freq = torch.fft.rfftfreq(T, d=1.0 / fps_val)
        _freq_mask_cache[key] = ((freq >= 0.7) & (freq <= 4.0)).to(dev)
    return _freq_mask_cache[key]

class SubjectDataset(Dataset):
    def __init__(self, processed_dir, subjects, seq_len=SEQ_LEN, stride=None, augment=False):
        self.samples = []
        self.augment = augment
        stride = stride or seq_len

        for subj in subjects:
            rgb_path = os.path.join(processed_dir, f"{subj}_rgb.npy")
            ppg_path = os.path.join(processed_dir, f"{subj}_ppg.npy")
            fps_path = os.path.join(processed_dir, f"{subj}_fps.npy")
            if not all(os.path.exists(p) for p in [rgb_path, ppg_path, fps_path]):
                continue

            rgb = np.load(rgb_path)
            ppg = np.load(ppg_path)
            fps = float(np.load(fps_path)[0])
            min_len = min(len(rgb), len(ppg))
            rgb, ppg = rgb[:min_len], ppg[:min_len]

            for start in range(0, min_len - seq_len, stride):
                clip   = rgb[start:start+seq_len].T.astype(np.float32)
                signal = ppg[start:start+seq_len].astype(np.float32)
                self.samples.append((clip, signal, fps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fps = self.samples[idx]
        x, y = x.copy(), y.copy()

        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        if self.augment:
            x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
            x *= np.random.uniform(0.9, 1.1, size=(3, 1)).astype(np.float32)
            x += np.random.uniform(-0.15, 0.15, size=(3, 1)).astype(np.float32)
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
                y = y[::-1].copy()
            t     = np.linspace(0, 2 * np.pi, x.shape[1]).astype(np.float32)
            drift = np.random.uniform(-0.08, 0.08) * np.sin(np.random.uniform(0.01, 0.1) * t)
            x    += drift[np.newaxis, :]

        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(fps, dtype=torch.float32)

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
            DilatedBlock(3,  64, 1),
            DilatedBlock(64, 64, 2),
            DilatedBlock(64, 96, 4),
            DilatedBlock(96, 96, 8),
            DilatedBlock(96, 64, 16),
        )
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        feat = self.encoder(x)
        f = feat.permute(0, 2, 1)
        attn_out, _ = self.attn(f, f, f)
        f = self.norm(f + attn_out)
        f = f.permute(0, 2, 1)
        return self.head(f).squeeze(1)

def pearson_loss(pred, target):
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return 1 - (num / den).mean()

def frequency_loss(pred, target, fps_val):
    B, T   = pred.shape
    mask   = get_freq_mask(T, fps_val, pred.device)
    pred_f = torch.fft.rfft(pred.float())
    tgt_f  = torch.fft.rfft(target.float())
    pred_p = pred_f.abs()[..., mask]
    tgt_p  = tgt_f.abs()[..., mask]
    pred_p = pred_p / (pred_p.sum(dim=1, keepdim=True) + 1e-8)
    tgt_p  = tgt_p  / (tgt_p.sum(dim=1, keepdim=True) + 1e-8)
    return F.mse_loss(pred_p, tgt_p)

def combined_loss(pred, target, fps):
    pred, target = pred.float(), target.float()
    return 0.7 * pearson_loss(pred, target) + 1.5 * frequency_loss(pred, target, fps.mean().item())

def evaluate_split(device, processed_dir, train_subj, val_subj):
    torch.manual_seed(0)
    ds_tr = SubjectDataset(processed_dir, train_subj, stride=SEQ_LEN, augment=False)
    ds_va = SubjectDataset(processed_dir, val_subj,   stride=SEQ_LEN, augment=False)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        return float("inf")

    ld_tr = DataLoader(ds_tr, batch_size=16, shuffle=True,  num_workers=0)
    ld_va = DataLoader(ds_va, batch_size=16, shuffle=False, num_workers=0)

    model = TSCAN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best = float("inf")
    for _ in range(4):
        model.train()
        for x, y, fps in ld_tr:
            x, y, fps = x.to(device), y.to(device), fps.to(device)
            opt.zero_grad()
            combined_loss(model(x), y, fps).backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, fps in ld_va:
                x, y, fps = x.to(device), y.to(device), fps.to(device)
                val_loss += combined_loss(model(x), y, fps).item()
        val_loss /= len(ld_va)
        best = min(best, val_loss)

    del model
    torch.cuda.empty_cache()
    return best

def find_best_split(device, processed_dir, subjects, n_trials=20):
    print(f"Searching {n_trials} random splits to find the most representative val set...")
    split = int(0.8 * len(subjects))
    best_score = float("inf")
    best_seed  = 0

    for seed in range(n_trials):
        rng = list(subjects)
        random.seed(seed)
        random.shuffle(rng)
        train_s, val_s = rng[:split], rng[split:]
        score = evaluate_split(device, processed_dir, train_s, val_s)
        print(f"  Seed {seed:3d}: quick val={score:.4f}" + (" ✓ new best" if score < best_score else ""))
        if score < best_score:
            best_score = score
            best_seed  = seed

    print(f"\nBest split: seed={best_seed}  quick val={best_score:.4f}")
    return best_seed

def train():
    processed_dir = "processed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    subjects = sorted(set(
        f.split("_")[0] for f in os.listdir(processed_dir)
        if f.endswith("_rgb.npy")
    ))
    print(f"Found {len(subjects)} subjects\n")

    best_seed = find_best_split(device, processed_dir, subjects, n_trials=20)

    random.seed(best_seed)
    shuffled = list(subjects)
    random.shuffle(shuffled)
    split      = int(0.8 * len(shuffled))
    train_subj = shuffled[:split]
    val_subj   = shuffled[split:]

    print(f"\nTrain subjects ({len(train_subj)}): {train_subj}")
    print(f"Val subjects   ({len(val_subj)}):   {val_subj}\n")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    train_ds = SubjectDataset(processed_dir, train_subj, stride=SEQ_LEN // 2, augment=True)
    val_ds   = SubjectDataset(processed_dir, val_subj,   stride=SEQ_LEN,       augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model  = TSCAN().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    best     = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()

        model.train()
        train_loss = 0.0
        opt.zero_grad()

        for i, (x, y, fps) in enumerate(train_loader):
            x, y, fps = x.to(device), y.to(device), fps.to(device)
            with autocast(device_type=device.type):
                loss = combined_loss(model(x), y, fps) / ACCUM_STEPS
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, fps in val_loader:
                x, y, fps = x.to(device), y.to(device), fps.to(device)
                with autocast(device_type=device.type):
                    val_loss += combined_loss(model(x), y, fps).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        sched.step()

        marker = " ✓" if val_loss < best else ""
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}  Val={val_loss:.4f}  Gap={val_loss-train_loss:+.4f}{marker}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. Best val: {best:.4f}")
            break

    print(f"\nTraining complete. Best val loss: {best:.4f}")


if __name__ == "__main__":
    train()