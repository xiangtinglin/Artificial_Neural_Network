#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GTZAN Kaggle Project - High-score training script (PANNs embedding + multi-crop + mixup + ensemble-ready)

Backend:
  - panns_embed (default): use AudioSet-pretrained PANNs to extract segment embeddings, then train a small classifier.
Why:
  - Fix the major bottleneck in the baseline (only ~3 seconds signal used) by multi-crop aggregation.
  - Transfer learning is the most reliable way to boost both public and private LB.

Outputs (out_dir):
  - checkpoint_best.pt
  - label_map.json
  - hparams.json
  - train_log.csv

This script expects CSV columns: ID,label,set
  - set in {train,val,test}
  - label available for train/val
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa is required. Please pip install librosa.") from e

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_label_map(df_train: "pd.DataFrame") -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({str(x).strip() for x in df_train["label"].dropna().tolist()})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label

def load_audio(path: str, sr: int) -> np.ndarray:
    # Mono float32 waveform
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        pad = target_len - len(y)
        return np.pad(y, (0, pad), mode="constant").astype(np.float32)
    return y[:target_len].astype(np.float32)

def crop_starts(num_samples: int, crop_len: int, n_crops: int, mode: str, rng: np.random.RandomState) -> np.ndarray:
    """
    mode:
      - uniform: equally spaced starts
      - random: random starts
    """
    if num_samples <= crop_len:
        return np.zeros((n_crops,), dtype=np.int64)

    max_start = num_samples - crop_len
    if mode == "uniform":
        if n_crops == 1:
            return np.array([max_start // 2], dtype=np.int64)
        return np.round(np.linspace(0, max_start, n_crops)).astype(np.int64)
    elif mode == "random":
        return rng.randint(0, max_start + 1, size=n_crops, dtype=np.int64)
    else:
        raise ValueError(f"Unknown crop mode: {mode}")

# -----------------------------
# PANNs embedding extractor
# -----------------------------
class PANNSEmbeddingExtractor:
    """
    Wrapper around panns-inference AudioTagging to output embeddings.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        try:
            from panns_inference import AudioTagging
        except Exception as e:
            raise RuntimeError(
                "panns-inference is required. Install with: pip install panns-inference"
            ) from e
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for reproducibility. Download a Cnn14*.pth file first.")
        self.device = device
        self.at = AudioTagging(checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def embed_batch(self, wave_batch: np.ndarray) -> np.ndarray:
        """
        wave_batch: (B, T) float32 waveform
        returns: (B, 2048) float32 embeddings
        """
        clipwise, emb = self.at.inference(wave_batch)
        # panns-inference may return torch.Tensor or np.ndarray depending on version
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

# -----------------------------
# Embedding cache builder
# -----------------------------
@dataclass
class CacheSpec:
    split_name: str
    crop_seconds: float
    n_crops: int
    crop_mode: str
    sample_rate: int
    seed: int

def cache_filename(spec: CacheSpec) -> str:
    s = f"{spec.split_name}_sr{spec.sample_rate}_crop{spec.crop_seconds:.2f}s_{spec.crop_mode}_n{spec.n_crops}_seed{spec.seed}.npz"
    return s.replace(".", "p")  # avoid extra dots

def build_or_load_cache(
    df: "pd.DataFrame",
    audio_root: str,
    label2idx: Optional[Dict[str, int]],
    extractor: PANNSEmbeddingExtractor,
    out_dir: str,
    split_name: str,
    sample_rate: int,
    crop_seconds: float,
    n_crops: int,
    crop_mode: str,
    seed: int,
    batch_embed: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
      X: (Nseg, D)
      y: (Nseg,) or missing if label2idx is None
      seg_audio_idx: (Nseg,) mapping segment -> audio index in audio_ids
      audio_ids: (Naudio,) original audio IDs
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = CacheSpec(split_name, crop_seconds, n_crops, crop_mode, sample_rate, seed)
    cpath = os.path.join(out_dir, cache_filename(spec))
    if os.path.isfile(cpath):
        data = np.load(cpath, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        return out

    df = df.reset_index(drop=True).copy()
    audio_ids = df["ID"].astype(str).str.strip().tolist()
    audio_paths = [os.path.join(audio_root, aid) for aid in audio_ids]

    naudio = len(audio_ids)
    crop_len = int(round(sample_rate * crop_seconds))

    seg_audio_idx: List[int] = []
    seg_labels: List[int] = []
    segments: List[np.ndarray] = []

    # Pre-load and segment
    for i, path in enumerate(audio_paths):
        if not os.path.isfile(path):
            # Keep determinism: treat missing as silence
            y = np.zeros((crop_len,), dtype=np.float32)
        else:
            y = load_audio(path, sr=sample_rate)
        rng = np.random.RandomState(seed + i * 10007)

        starts = crop_starts(len(y), crop_len, n_crops, crop_mode, rng)
        for st in starts:
            seg = pad_or_trim(y[st: st + crop_len], crop_len)
            segments.append(seg)
            seg_audio_idx.append(i)
            if label2idx is not None:
                lab = str(df.loc[i, "label"]).strip()
                if lab not in label2idx:
                    raise ValueError(f"Label '{lab}' not found in label2idx.")
                seg_labels.append(label2idx[lab])

    # Batch embed
    X_list: List[np.ndarray] = []
    for j in range(0, len(segments), batch_embed):
        batch = np.stack(segments[j:j + batch_embed], axis=0).astype(np.float32)
        emb = extractor.embed_batch(batch)
        X_list.append(emb)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    seg_audio_idx_arr = np.asarray(seg_audio_idx, dtype=np.int64)

    out: Dict[str, np.ndarray] = {
        "X": X,
        "seg_audio_idx": seg_audio_idx_arr,
        "audio_ids": np.asarray(audio_ids, dtype=object),
    }
    if label2idx is not None:
        out["y"] = np.asarray(seg_labels, dtype=np.int64)

    np.savez_compressed(cpath, **out)
    return out

# -----------------------------
# Classifier (small MLP on embeddings)
# -----------------------------
class EmbedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------
# Dataset / Loader
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x_mix = lam * x + (1 - lam) * x2
    return x_mix, y, y2, lam

@torch.no_grad()
def eval_clipwise(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    seg_audio_idx: np.ndarray,
    num_audio: int,
    device: str,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """
    Aggregate segment logits to clip logits by mean, then compute loss & accuracy on clips.
    """
    model.eval()
    logits_sum = np.zeros((num_audio, model.net[-1].out_features), dtype=np.float64)
    counts = np.zeros((num_audio,), dtype=np.int64)

    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        lb = model(xb).detach().cpu().numpy()  # (B, C)
        idxb = seg_audio_idx[i:i + batch_size]
        for k, aidx in enumerate(idxb):
            logits_sum[aidx] += lb[k]
            counts[aidx] += 1

    counts = np.maximum(counts, 1)
    clip_logits = (logits_sum / counts[:, None]).astype(np.float32)
    pred = np.argmax(clip_logits, axis=1)
    acc = float(np.mean(pred == y)) * 100.0

    # clipwise loss
    clip_logits_t = torch.from_numpy(clip_logits)
    y_t = torch.from_numpy(y)
    loss_fn = nn.CrossEntropyLoss()
    loss = float(loss_fn(clip_logits_t, y_t).item())
    return loss, acc

# -----------------------------
# Main
# -----------------------------
@dataclass
class HParams:
    backend: str
    sample_rate: int
    crop_seconds: float
    train_crops: int
    val_crops: int
    crop_mode_train: str
    crop_mode_eval: str
    panns_checkpoint: str

    embed_dim: int
    clf_hidden: int
    clf_dropout: float

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    mixup_alpha: float

    seed: int
    num_workers: int

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="checkpoint_panns")

    ap.add_argument("--backend", type=str, default="panns_embed", choices=["panns_embed"])

    # PANNs
    ap.add_argument("--panns_checkpoint", type=str, required=True, help="Path to Cnn14*.pth checkpoint")
    ap.add_argument("--sample_rate", type=int, default=32000)
    ap.add_argument("--crop_seconds", type=float, default=10.0)
    ap.add_argument("--train_crops", type=int, default=12)
    ap.add_argument("--val_crops", type=int, default=9)
    ap.add_argument("--crop_mode_train", type=str, default="random", choices=["random"])
    ap.add_argument("--crop_mode_eval", type=str, default="uniform", choices=["uniform"])

    # classifier
    ap.add_argument("--embed_dim", type=int, default=2048)
    ap.add_argument("--clf_hidden", type=int, default=512)
    ap.add_argument("--clf_dropout", type=float, default=0.4)

    # optim
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup_alpha", type=float, default=0.4)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip()

    train_df = df[df["set"] == "train"].copy()
    val_df = df[df["set"].isin(["val", "dev", "valid", "validation"])].copy()
    test_df = df[df["set"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Need both train and val rows in CSV for stable model selection.")

    label2idx, idx2label = build_label_map(train_df)
    num_classes = len(label2idx)

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, ensure_ascii=False, indent=2)

    hp = HParams(
        backend=args.backend,
        sample_rate=args.sample_rate,
        crop_seconds=args.crop_seconds,
        train_crops=args.train_crops,
        val_crops=args.val_crops,
        crop_mode_train=args.crop_mode_train,
        crop_mode_eval=args.crop_mode_eval,
        panns_checkpoint=args.panns_checkpoint,
        embed_dim=args.embed_dim,
        clf_hidden=args.clf_hidden,
        clf_dropout=args.clf_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    with open(os.path.join(args.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(hp) | {"num_classes": num_classes}, f, ensure_ascii=False, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Classes] {num_classes}: {list(label2idx.keys())}")

    extractor = PANNSEmbeddingExtractor(checkpoint_path=args.panns_checkpoint, device=device)

    cache_dir = os.path.join(args.out_dir, "cache")
    train_cache = build_or_load_cache(
        train_df, args.audio_root, label2idx, extractor, cache_dir, "train",
        args.sample_rate, args.crop_seconds, args.train_crops, args.crop_mode_train, args.seed,
        batch_embed=max(8, args.batch_size // 8),
    )
    val_cache = build_or_load_cache(
        val_df, args.audio_root, label2idx, extractor, cache_dir, "val",
        args.sample_rate, args.crop_seconds, args.val_crops, args.crop_mode_eval, args.seed,
        batch_embed=max(8, args.batch_size // 8),
    )

    Xtr, ytr = train_cache["X"], train_cache["y"]
    Xva, yva = val_cache["X"], val_cache["y"]
    seg_va = val_cache["seg_audio_idx"]
    num_va_audio = len(val_cache["audio_ids"])

    # For clipwise eval, we need y per audio (not per segment)
    # val_cache stores y per segment; fold it to audio-level by taking the first segment's label per audio
    y_va_audio = np.zeros((num_va_audio,), dtype=np.int64)
    seen = set()
    for seg_i, aidx in enumerate(seg_va):
        if aidx not in seen:
            y_va_audio[aidx] = int(yva[seg_i])
            seen.add(int(aidx))

    train_ds = EmbeddingDataset(Xtr, ytr)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )

    model = EmbedMLP(args.embed_dim, args.clf_hidden, num_classes, args.clf_dropout).to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, "checkpoint_best.pt")

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,sec\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            xb, ya, yb2, lam = mixup_batch(xb, yb, args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            loss = lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            running_acc += float((pred == ya).float().sum().item())  # approximate (mixup)
            n += xb.size(0)

        scheduler.step()
        train_loss = running_loss / max(n, 1)
        train_acc = (running_acc / max(n, 1)) * 100.0

        # clipwise val
        val_loss, val_acc = eval_clipwise(
            model, Xva, y_va_audio, seg_va, num_va_audio, device=device, batch_size=args.batch_size
        )

        sec = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc~={train_acc:.2f} | val_loss={val_loss:.4f} val_acc={val_acc:.2f} | lr={lr_now:.2e}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f},{lr_now:.8e},{sec:.2f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "backend": "panns_embed",
                    "model_state_dict": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "hparams": asdict(hp) | {"num_classes": num_classes},
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"[BEST] saved -> {best_path}")

    print(f"[Done] Best val acc: {best_val_acc:.2f} | ckpt: {best_path}")
    print(f"[Note] test rows in CSV = {len(test_df)} (test labels are hidden).")

if __name__ == "__main__":
    main()
