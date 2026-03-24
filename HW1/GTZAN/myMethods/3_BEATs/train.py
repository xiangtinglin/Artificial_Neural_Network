#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GTZAN Kaggle Project - BEATs embedding training script

Design goal:
  - Keep the same stable recipe as the user's PANNs version:
      * multi-crop training/eval
      * frozen pretrained audio backbone
      * small classifier head
      * mixup + label smoothing
      * clip-level logit averaging for validation model selection
  - Only replace the embedding extractor with BEATs.

Expected CSV columns:
  ID,label,set
where set in {train,val,test} (also accepts dev/valid/validation for val).

Requirements:
  1) Clone Microsoft BEATs / unilm repo, or copy its beats/ folder locally.
  2) Make sure Python can import BEATs.py and backbone.py.
  3) Provide a BEATs checkpoint path, e.g. BEATs_iter3+ AS2M.

Example:
  python train.py \
    --csv_path /path/to/gtzan.csv \
    --audio_root /path/to/genres \
    --out_dir /path/to/checkpoints_beats_seed42 \
    --beats_checkpoint /path/to/BEATs_iter3_plus_AS2M.pt \
    --beats_repo_dir /path/to/unilm/beats \
    --sample_rate 16000 \
    --crop_seconds 10.0 \
    --train_crops 12 \
    --val_crops 9 \
    --batch_size 128 \
    --epochs 60 \
    --lr 2e-4 \
    --weight_decay 1e-2 \
    --label_smoothing 0.05 \
    --mixup_alpha 0.2 \
    --clf_hidden 768 \
    --clf_dropout 0.3 \
    --seed 42
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa is required. Please pip install librosa.") from e


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
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
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def crop_starts(
    num_samples: int,
    crop_len: int,
    n_crops: int,
    mode: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    if num_samples <= crop_len:
        return np.zeros((n_crops,), dtype=np.int64)

    max_start = num_samples - crop_len
    if mode == "uniform":
        if n_crops == 1:
            return np.array([max_start // 2], dtype=np.int64)
        return np.round(np.linspace(0, max_start, n_crops)).astype(np.int64)
    if mode == "random":
        return rng.randint(0, max_start + 1, size=n_crops, dtype=np.int64)
    raise ValueError(f"Unknown crop mode: {mode}")


def resolve_audio_path(audio_root: str, track_id: str) -> str:
    track_id = str(track_id).strip()
    direct = os.path.join(audio_root, track_id)
    if os.path.isfile(direct):
        return direct

    stem, ext = os.path.splitext(track_id)
    if ext == "":
        direct2 = os.path.join(audio_root, track_id + ".au")
        if os.path.isfile(direct2):
            return direct2

    # GTZAN common layout: audio_root/<label>/<ID>
    for root, _, files in os.walk(audio_root):
        if track_id in files:
            return os.path.join(root, track_id)
        if ext == "" and (track_id + ".au") in files:
            return os.path.join(root, track_id + ".au")

    raise FileNotFoundError(f"Audio file not found for ID={track_id} under {audio_root}")


# -----------------------------
# Dataset
# -----------------------------
class SegmentEmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# -----------------------------
# BEATs extractor
# -----------------------------
class BEATSEmbeddingExtractor:
    def __init__(self, checkpoint_path: str, repo_dir: Optional[str], device: str = "cuda"):
        self.device = torch.device(device)

        if repo_dir is not None and repo_dir.strip() != "":
            repo_dir = os.path.abspath(repo_dir)
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)

        try:
            from BEATs import BEATs, BEATsConfig
        except Exception as e:
            raise ImportError(
                "Cannot import BEATs. Please clone microsoft/unilm/beats and pass --beats_repo_dir to that folder."
            ) from e

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.to(self.device)

        self.model = model
        self.embed_dim = int(cfg.encoder_embed_dim)

    @torch.no_grad()
    def extract_batch(self, wav_batch: np.ndarray) -> np.ndarray:
        """
        Input:
            wav_batch: (B, T) float32, 16kHz mono, already padded/truncated to same length.
        Output:
            embeddings: (B, 2D) where 2D = mean + std pooling over time.
        """
        x = torch.from_numpy(wav_batch).to(self.device, non_blocking=True)
        padding_mask = torch.zeros(x.shape, dtype=torch.bool, device=self.device)

        out = self.model.extract_features(x, padding_mask=padding_mask)
        if isinstance(out, tuple):
            feats = out[0]
        else:
            feats = out

        # feats: (B, T', D)
        mean_pool = feats.mean(dim=1)
        std_pool = feats.std(dim=1, unbiased=False)
        emb = torch.cat([mean_pool, std_pool], dim=1)
        return emb.detach().cpu().numpy().astype(np.float32)


# -----------------------------
# Feature extraction
# -----------------------------
@torch.no_grad()
def build_segment_table(
    df_split: pd.DataFrame,
    label2idx: Dict[str, int],
    audio_root: str,
    sample_rate: int,
    crop_seconds: float,
    n_crops: int,
    crop_mode: str,
    extractor: BEATSEmbeddingExtractor,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns:
        X           : (num_segments, feat_dim)
        y_audio     : (num_audio,)
        seg_audio_i : (num_segments,) mapping each segment to its clip index
        num_audio   : int
    """
    crop_len = int(round(sample_rate * crop_seconds))
    rng = np.random.RandomState(seed)

    segments: List[np.ndarray] = []
    seg_audio_idx: List[int] = []
    y_audio: List[int] = []

    for audio_i, row in enumerate(df_split.itertuples(index=False)):
        audio_id = str(getattr(row, "ID")).strip()
        label = str(getattr(row, "label")).strip() if hasattr(row, "label") else None
        path = resolve_audio_path(audio_root, audio_id)

        y = load_audio(path, sr=sample_rate)
        starts = crop_starts(len(y), crop_len, n_crops, crop_mode, rng)

        for s in starts:
            seg = pad_or_trim(y[int(s): int(s) + crop_len], crop_len)
            segments.append(seg)
            seg_audio_idx.append(audio_i)

        if label is not None and label != "nan":
            y_audio.append(label2idx[label])

    if len(segments) == 0:
        raise ValueError("No audio segments were created.")

    wavs = np.stack(segments, axis=0).astype(np.float32)
    feat_chunks: List[np.ndarray] = []
    for i in range(0, len(wavs), batch_size):
        feat_chunks.append(extractor.extract_batch(wavs[i:i + batch_size]))
    X = np.concatenate(feat_chunks, axis=0).astype(np.float32)

    y_audio_np = np.array(y_audio, dtype=np.int64)
    seg_audio_idx_np = np.array(seg_audio_idx, dtype=np.int64)
    num_audio = len(df_split)
    return X, y_audio_np, seg_audio_idx_np, num_audio


# -----------------------------
# Model
# -----------------------------
class MLPClassifier(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Train / Eval helpers
# -----------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x_mix = lam * x + (1.0 - lam) * x2
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
    model.eval()
    num_classes = model.net[-1].out_features

    logits_sum = np.zeros((num_audio, num_classes), dtype=np.float64)
    counts = np.zeros((num_audio,), dtype=np.int64)

    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device, non_blocking=True)
        lb = model(xb).detach().cpu().numpy()
        idxb = seg_audio_idx[i:i + batch_size]
        for k, aidx in enumerate(idxb):
            logits_sum[aidx] += lb[k]
            counts[aidx] += 1

    counts = np.maximum(counts, 1)
    clip_logits = (logits_sum / counts[:, None]).astype(np.float32)
    pred = np.argmax(clip_logits, axis=1)
    acc = float(np.mean(pred == y)) * 100.0

    clip_logits_t = torch.from_numpy(clip_logits)
    y_t = torch.from_numpy(y)
    loss = float(nn.CrossEntropyLoss()(clip_logits_t, y_t).item())
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
    beats_checkpoint: str
    beats_repo_dir: str
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
    ap.add_argument("--out_dir", type=str, default="checkpoint_beats")

    ap.add_argument("--backend", type=str, default="beats_embed", choices=["beats_embed"])

    # BEATs
    ap.add_argument("--beats_checkpoint", type=str, required=True, help="Path to BEATs*.pt checkpoint")
    ap.add_argument(
        "--beats_repo_dir",
        type=str,
        required=True,
        help="Path to cloned unilm/beats directory containing BEATs.py and backbone.py",
    )
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--crop_seconds", type=float, default=10.0)
    ap.add_argument("--train_crops", type=int, default=12)
    ap.add_argument("--val_crops", type=int, default=9)
    ap.add_argument("--crop_mode_train", type=str, default="random", choices=["random"])
    ap.add_argument("--crop_mode_eval", type=str, default="uniform", choices=["uniform"])

    # classifier
    ap.add_argument(
        "--embed_dim",
        type=int,
        default=1536,
        help="Final embedding dim after pooling. For BEATs base (768), mean+std => 1536.",
    )
    ap.add_argument("--clf_hidden", type=int, default=768)
    ap.add_argument("--clf_dropout", type=float, default=0.3)

    # optimization
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()
    else:
        df["label"] = ""

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
        beats_checkpoint=args.beats_checkpoint,
        beats_repo_dir=args.beats_repo_dir,
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

    extractor = BEATSEmbeddingExtractor(
        checkpoint_path=args.beats_checkpoint,
        repo_dir=args.beats_repo_dir,
        device=device,
    )
    inferred_embed_dim = extractor.embed_dim * 2
    if args.embed_dim != inferred_embed_dim:
        print(f"[Warn] --embed_dim={args.embed_dim} but BEATs mean+std pooling gives {inferred_embed_dim}. Using inferred value.")
        args.embed_dim = inferred_embed_dim

    print("[1/3] Extracting train embeddings...")
    Xtr, y_tr_audio, seg_tr, num_tr_audio = build_segment_table(
        train_df,
        label2idx,
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        crop_seconds=args.crop_seconds,
        n_crops=args.train_crops,
        crop_mode=args.crop_mode_train,
        extractor=extractor,
        batch_size=max(1, min(args.batch_size, 64)),
        seed=args.seed,
    )
    y_tr_seg = y_tr_audio[seg_tr]
    print(f"  train segments: {Xtr.shape}, train clips: {num_tr_audio}")

    print("[2/3] Extracting val embeddings...")
    Xva, y_va_audio, seg_va, num_va_audio = build_segment_table(
        val_df,
        label2idx,
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        crop_seconds=args.crop_seconds,
        n_crops=args.val_crops,
        crop_mode=args.crop_mode_eval,
        extractor=extractor,
        batch_size=max(1, min(args.batch_size, 64)),
        seed=args.seed + 123,
    )
    print(f"  val segments: {Xva.shape}, val clips: {num_va_audio}")
    print(f"  test rows in CSV (hidden labels): {len(test_df)}")

    train_ds = SegmentEmbeddingDataset(Xtr, y_tr_seg)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = MLPClassifier(
        embed_dim=args.embed_dim,
        hidden_dim=args.clf_hidden,
        dropout=args.clf_dropout,
        num_classes=num_classes,
    ).to(device)

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
            loss = lam * loss_fn(logits, ya) + (1.0 - lam) * loss_fn(logits, yb2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            running_acc += float((pred == ya).float().sum().item())
            n += xb.size(0)

        scheduler.step()
        train_loss = running_loss / max(n, 1)
        train_acc = (running_acc / max(n, 1)) * 100.0

        val_loss, val_acc = eval_clipwise(
            model=model,
            X=Xva,
            y=y_va_audio,
            seg_audio_idx=seg_va,
            num_audio=num_va_audio,
            device=device,
            batch_size=args.batch_size,
        )

        sec = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc~={train_acc:.2f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} | lr={lr_now:.2e}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f},{lr_now:.8e},{sec:.2f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "backend": "beats_embed",
                    "model_state_dict": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "hparams": asdict(hp) | {"num_classes": num_classes, "embed_dim": args.embed_dim},
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