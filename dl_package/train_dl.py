"""
train_dl.py - train a small U-Net that finds walls in the scan image layers.

  python train_dl.py --data data --epochs 80

Classes: 0 background | 1 wall | 2 low wall | 3 zellige.   Label 255 = not drawn (ignored).
The right-most part of the labelled area (--val-frac) is held out and never used for training, so the
printed F1 is measured on ground the network has not seen.
Works on CPU or an NVIDIA GPU (auto-detected, mixed precision on GPU).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, cin, ncls=4, base=24, depth=4):
        super().__init__()
        ch = [base * 2 ** k for k in range(depth + 1)]
        self.enc = nn.ModuleList([Block(cin if k == 0 else ch[k - 1], ch[k]) for k in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[k + 1], ch[k], 2, stride=2) for k in range(depth)])
        self.dec = nn.ModuleList([Block(ch[k] * 2, ch[k]) for k in range(depth)])
        self.head = nn.Conv2d(ch[0], ncls, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for k in range(self.depth):
            x = self.enc[k](x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.enc[self.depth](x)
        for k in reversed(range(self.depth)):
            x = self.up[k](x)
            x = self.dec[k](torch.cat([x, skips[k]], 1))
        return self.head(x)


# --------------------------------------------------------------------------- data
def load(data_dir):
    stack = np.load(os.path.join(data_dir, "stack.npz"))["stack"]              # (C,H,W) float16
    labels = np.load(os.path.join(data_dir, "labels.npz"))["labels"]           # (H,W) uint8
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    return stack, labels, meta


def channel_stats(stack):
    valid = stack[-1] > 0
    mean = np.array([float(stack[c][valid].mean()) for c in range(stack.shape[0])], np.float32)
    std = np.array([float(stack[c][valid].std()) + 1e-3 for c in range(stack.shape[0])], np.float32)
    mean[-1], std[-1] = 0.0, 1.0                       # the 'valid' channel stays 0/1
    return mean, std


def split_regions(labels, val_frac):
    sup = labels != 255
    cols = np.nonzero(sup.any(0))[0]
    cut = int(np.quantile(cols, 1.0 - val_frac))
    train = sup.copy()
    train[:, cut:] = False
    val = sup.copy()
    val[:, :cut] = False
    return train, val, cut


def crop(stack, labels, r, c, T, region=None):
    H, W = labels.shape
    r0, c0 = max(0, min(H - T, r - T // 2)), max(0, min(W - T, c - T // 2))
    x = stack[:, r0:r0 + T, c0:c0 + T]
    y = labels[r0:r0 + T, c0:c0 + T].copy()
    if region is not None:                                      # outside this region the label does not count
        y[~region[r0:r0 + T, c0:c0 + T]] = 255
    return x, y


def augment(x, y, rng):
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if rng.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    k = rng.randrange(4)
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    x = np.ascontiguousarray(x).astype(np.float32)
    if rng.random() < 0.15:                                      # sometimes hide the colour: stay robust to bad colours
        x[5:8] = 0
    x[5:8] += np.random.normal(0, 0.03, x[5:8].shape).astype(np.float32)
    return x, np.ascontiguousarray(y)


class Sampler:
    def __init__(self, stack, labels, region, T, mean, std, seed=0):
        self.stack, self.labels, self.region, self.T = stack, labels, region, T
        self.mean, self.std = mean[:, None, None], std[:, None, None]
        self.rng = random.Random(seed)
        pos = np.argwhere((labels > 0) & (labels < 255) & region)
        anyl = np.argwhere(region)
        self.pos, self.any = pos[::7], anyl[::97]
        if len(self.any) == 0:
            raise SystemExit("no labelled cells in the training region")

    def batch(self, n, device):
        xs, ys = [], []
        for _ in range(n):
            pool = self.pos if (len(self.pos) and self.rng.random() < 0.6) else self.any
            r, c = pool[self.rng.randrange(len(pool))]
            r += self.rng.randint(-self.T // 3, self.T // 3)
            c += self.rng.randint(-self.T // 3, self.T // 3)
            x, y = crop(self.stack, self.labels, int(r), int(c), self.T, self.region)
            x = (x.astype(np.float32) - self.mean) / self.std
            x, y = augment(x, y, self.rng)
            xs.append(x)
            ys.append(y)
        return (torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).long().to(device))


# --------------------------------------------------------------------------- loss / metrics
def dice_loss(logits, y, cls=1):
    p = torch.softmax(logits, 1)[:, cls]
    m = (y != 255)
    g = (y == cls) & m
    p = p * m
    inter = (p * g).sum()
    return 1 - (2 * inter + 1) / (p.sum() + g.sum() + 1)


@torch.no_grad()
def predict_full(model, stack, mean, std, device, tile=512, overlap=64):
    C, H, W = stack.shape
    out = torch.zeros(model.head.out_channels, H, W)
    cnt = torch.zeros(1, H, W)
    step = tile - overlap
    mean_t = torch.from_numpy(mean)[:, None, None]
    std_t = torch.from_numpy(std)[:, None, None]
    for r in range(0, max(H - overlap, 1), step):
        for c in range(0, max(W - overlap, 1), step):
            r0, c0 = min(r, max(H - tile, 0)), min(c, max(W - tile, 0))
            x = torch.from_numpy(stack[:, r0:r0 + tile, c0:c0 + tile].astype(np.float32))
            x = ((x - mean_t) / std_t).unsqueeze(0).to(device)
            h, w = x.shape[2:]
            ph, pw = (-h) % 16, (-w) % 16
            x = F.pad(x, (0, pw, 0, ph))
            p = torch.softmax(model(x), 1)[0, :, :h, :w].cpu()
            out[:, r0:r0 + h, c0:c0 + w] += p
            cnt[:, r0:r0 + h, c0:c0 + w] += 1
    return out / cnt.clamp(min=1)


def wall_f1(prob, labels, region, tol=2):
    """F1 of the wall class with a tolerance of `tol` cells (10 cm each), over the labelled held-out area"""
    pred = (prob.argmax(0) == 1)
    gt = torch.from_numpy(labels == 1)
    m = torch.from_numpy(region)
    pred, gt = pred & m, gt & m

    def dil(a):
        return F.max_pool2d(a.float()[None, None], 2 * tol + 1, 1, tol)[0, 0] > 0
    tp_p = (pred & dil(gt)).sum().item()
    tp_r = (gt & dil(pred)).sum().item()
    prec = tp_p / max(pred.sum().item(), 1)
    rec = tp_r / max(gt.sum().item(), 1)
    return (2 * prec * rec / max(prec + rec, 1e-9)), prec, rec


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps", type=int, default=100, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--base", type=int, default=24, help="U-Net width (use 32 on a good GPU)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)
    stack, labels, meta = load(a.data)
    mean, std = channel_stats(stack)
    train_r, val_r, cut = split_regions(labels, a.val_frac)
    print(f"stack {stack.shape}, labelled cells {int((labels != 255).sum()):,} | train {int(train_r.sum()):,} | "
          f"held-out {int(val_r.sum()):,} (columns >= {cut})", flush=True)
    if val_r.sum() < 5000:
        print("WARNING: very little held-out labelled area; the F1 below is not reliable")
    sampler = Sampler(stack, labels, train_r, a.tile, mean, std, a.seed)

    model = UNet(stack.shape[0], 4, a.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    weights = torch.tensor([1.0, 8.0, 8.0, 4.0], device=dev)
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for _ in range(a.steps):
            x, y = sampler.batch(a.batch, dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                lo = model(x)
                loss = F.cross_entropy(lo, y, weight=weights, ignore_index=255) + 0.5 * dice_loss(lo.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        if ep % 2 == 0 or ep == a.epochs:
            model.eval()
            prob = predict_full(model, stack, mean, std, dev)
            f1, p, r = wall_f1(prob, labels, val_r)
            flag = ""
            if f1 > best:
                best = f1
                torch.save({"state": model.state_dict(), "cfg": {"cin": stack.shape[0], "ncls": 4, "base": a.base},
                            "mean": mean, "std": std, "meta": meta, "val_f1": f1}, a.out)
                flag = "  <- saved"
            print(f"epoch {ep:3d}  loss {tot / a.steps:.4f}  held-out wall F1 {f1:.3f} (precision {p:.3f}, recall {r:.3f})"
                  f"  {time.time() - t0:.0f}s{flag}", flush=True)
        else:
            print(f"epoch {ep:3d}  loss {tot / a.steps:.4f}  {time.time() - t0:.0f}s", flush=True)
    print(f"best held-out wall F1: {best:.3f}  -> {a.out}")


if __name__ == "__main__":
    main()
