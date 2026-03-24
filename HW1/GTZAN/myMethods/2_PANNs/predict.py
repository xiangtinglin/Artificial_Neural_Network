#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GTZAN Kaggle Project - High-score inference script (PANNs embedding + multi-crop + checkpoint ensemble)

Input:
  --csv_path: CSV with ID,label,set
  --audio_root: folder containing audio files
  --checkpoint_path OR --checkpoint_paths: trained classifier ckpt(s)
Output:
  --out_csv: submission file in format:
      ID,label
      00686.au,jazz
      ...

This script computes PANNs embeddings on multiple crops per audio and averages logits.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa is required. Please pip install librosa.") from e

# -----------------------------
# shared helpers
# -----------------------------
def load_audio(path: str, sr: int) -> np.ndarray:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant").astype(np.float32)
    return y[:target_len].astype(np.float32)

def crop_starts_uniform(num_samples: int, crop_len: int, n_crops: int) -> np.ndarray:
    if num_samples <= crop_len:
        return np.zeros((n_crops,), dtype=np.int64)
    max_start = num_samples - crop_len
    if n_crops == 1:
        return np.array([max_start // 2], dtype=np.int64)
    return np.round(np.linspace(0, max_start, n_crops)).astype(np.int64)

class PANNSEmbeddingExtractor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        try:
            from panns_inference import AudioTagging
        except Exception as e:
            raise RuntimeError("panns-inference is required. Install with: pip install panns-inference") from e
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for reproducibility.")
        self.device = device
        self.at = AudioTagging(checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def embed_batch(self, wave_batch: np.ndarray) -> np.ndarray:
        _, emb = self.at.inference(wave_batch)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

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

def load_ckpt(path: str, device: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("backend") != "panns_embed":
        raise ValueError(f"Unsupported backend in ckpt: {ckpt.get('backend')}")
    return ckpt

def build_model_from_ckpt(ckpt: Dict[str, Any], device: str) -> nn.Module:
    hp = ckpt["hparams"]
    model = EmbedMLP(
        in_dim=int(hp["embed_dim"]),
        hidden_dim=int(hp["clf_hidden"]),
        num_classes=int(hp["num_classes"]),
        dropout=float(hp["clf_dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model

# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)

    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--checkpoint_paths", type=str, default="", help="comma-separated list of ckpt paths (ensemble)")

    ap.add_argument("--out_csv", type=str, default="submission.csv")
    ap.add_argument("--tta_crops", type=int, default=15, help="number of uniform crops per audio for TTA")
    ap.add_argument("--embed_batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ckpt list
    ckpt_paths: List[str] = []
    if args.checkpoint_paths.strip():
        ckpt_paths = [p.strip() for p in args.checkpoint_paths.split(",") if p.strip()]
    elif args.checkpoint_path.strip():
        ckpt_paths = [args.checkpoint_path.strip()]
    else:
        raise ValueError("Provide --checkpoint_path or --checkpoint_paths")

    # load first ckpt to get label map / hparams
    ckpt0 = load_ckpt(ckpt_paths[0], device=device)
    label2idx = ckpt0["label2idx"]
    idx2label = ckpt0["idx2label"]
    hp0 = ckpt0["hparams"]

    sample_rate = int(hp0["sample_rate"])
    crop_seconds = float(hp0["crop_seconds"])
    crop_len = int(round(sample_rate * crop_seconds))

    panns_ckpt = str(hp0["panns_checkpoint"])
    extractor = PANNSEmbeddingExtractor(checkpoint_path=panns_ckpt, device=device)

    models = []
    for p in ckpt_paths:
        ck = load_ckpt(p, device=device)
        # safety: same extractor config
        hp = ck["hparams"]
        if str(hp["panns_checkpoint"]) != panns_ckpt or int(hp["sample_rate"]) != sample_rate or float(hp["crop_seconds"]) != crop_seconds:
            raise ValueError("All ckpts in ensemble must share same PANNs checkpoint/sample_rate/crop_seconds.")
        models.append(build_model_from_ckpt(ck, device=device))

    # read CSV
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()

    test_df = df[df["set"] == "test"].copy().reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError("No test rows found (set=test).")

    # build all segments for test
    seg_audio_idx: List[int] = []
    segments: List[np.ndarray] = []
    audio_ids = test_df["ID"].tolist()

    for aidx, aid in enumerate(audio_ids):
        path = os.path.join(args.audio_root, aid)
        if not os.path.isfile(path):
            y = np.zeros((crop_len,), dtype=np.float32)
        else:
            y = load_audio(path, sr=sample_rate)

        starts = crop_starts_uniform(len(y), crop_len, args.tta_crops)
        for st in starts:
            seg = pad_or_trim(y[st: st + crop_len], crop_len)
            segments.append(seg)
            seg_audio_idx.append(aidx)

    # embed all segments
    X_list: List[np.ndarray] = []
    for i in range(0, len(segments), args.embed_batch):
        batch = np.stack(segments[i:i + args.embed_batch], axis=0).astype(np.float32)
        emb = extractor.embed_batch(batch)
        X_list.append(emb)
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    seg_audio_idx_arr = np.asarray(seg_audio_idx, dtype=np.int64)

    # ensemble logits
    num_audio = len(audio_ids)
    num_classes = len(label2idx)
    logits_sum = np.zeros((num_audio, num_classes), dtype=np.float64)
    counts = np.zeros((num_audio,), dtype=np.int64)

    X_t = torch.from_numpy(X).to(device)
    for mdl in models:
        with torch.no_grad():
            logits_seg = mdl(X_t).detach().cpu().numpy()  # (Nseg, C)
        # accumulate by audio
        for k, aidx in enumerate(seg_audio_idx_arr):
            logits_sum[aidx] += logits_seg[k]

    # each model contributed args.tta_crops segs per audio
    counts[:] = len(models) * args.tta_crops
    logits_audio = (logits_sum / counts[:, None]).astype(np.float32)
    pred_idx = np.argmax(logits_audio, axis=1)
    pred_label = [idx2label[int(i)] for i in pred_idx]

    out = pd.DataFrame({"ID": audio_ids, "label": pred_label})
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[Done] wrote {args.out_csv} | rows={len(out)}")

if __name__ == "__main__":
    main()
