#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import librosa

# -----------------------------
# utils（完全沿用）
# -----------------------------
def load_audio(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def crop_starts_uniform(n, crop_len, n_crops):
    if n <= crop_len:
        return np.zeros(n_crops, dtype=np.int64)
    max_start = n - crop_len
    return np.round(np.linspace(0, max_start, n_crops)).astype(np.int64)

# -----------------------------
# BEATs extractor
# -----------------------------
class BEATSEmbeddingExtractor:
    def __init__(self, ckpt_path, repo_dir, device="cuda"):
        sys.path.insert(0, repo_dir)

        from BEATs import BEATs, BEATsConfig

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])

        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval().to(device)

        self.model = model
        self.device = device

    @torch.no_grad()
    def embed_batch(self, wav_batch):
        x = torch.from_numpy(wav_batch).to(self.device)
        padding_mask = torch.zeros(x.shape, dtype=torch.bool).to(self.device)

        feats, _ = self.model.extract_features(x, padding_mask=padding_mask)

        mean = feats.mean(dim=1)
        std = feats.std(dim=1)

        emb = torch.cat([mean, std], dim=1)
        return emb.cpu().numpy()

# -----------------------------
# classifier（同你 train）
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    if ckpt["backend"] != "beats_embed":
        raise ValueError("Not BEATs ckpt")
    return ckpt

def build_model(ckpt, device):
    hp = ckpt["hparams"]

    model = MLP(
        in_dim=hp["embed_dim"],
        hidden=hp["clf_hidden"],
        num_classes=hp["num_classes"],
        dropout=hp["clf_dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--audio_root", required=True)

    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--checkpoint_paths", default="")

    parser.add_argument("--out_csv", default="submission.csv")
    parser.add_argument("--tta_crops", type=int, default=21)
    parser.add_argument("--embed_batch", type=int, default=32)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ckpt list
    if args.checkpoint_paths:
        paths = args.checkpoint_paths.split(",")
    else:
        paths = [args.checkpoint_path]

    ckpt0 = load_ckpt(paths[0], device)

    label2idx = ckpt0["label2idx"]
    idx2label = ckpt0["idx2label"]
    hp = ckpt0["hparams"]

    sr = hp["sample_rate"]
    crop_len = int(sr * hp["crop_seconds"])

    extractor = BEATSEmbeddingExtractor(
        ckpt_path=hp["beats_checkpoint"],
        repo_dir=hp["beats_repo_dir"],
        device=device,
    )

    models = [build_model(load_ckpt(p, device), device) for p in paths]

    # read test
    df = pd.read_csv(args.csv_path)
    df["set"] = df["set"].str.lower()
    test_df = df[df["set"] == "test"]

    audio_ids = test_df["ID"].tolist()

    segments = []
    seg_idx = []

    for i, aid in enumerate(audio_ids):
        path = os.path.join(args.audio_root, aid)

        if not os.path.exists(path):
            y = np.zeros(crop_len)
        else:
            y = load_audio(path, sr)

        starts = crop_starts_uniform(len(y), crop_len, args.tta_crops)

        for s in starts:
            seg = pad_or_trim(y[s:s+crop_len], crop_len)
            segments.append(seg)
            seg_idx.append(i)

    # embedding
    X = []
    for i in range(0, len(segments), args.embed_batch):
        batch = np.stack(segments[i:i+args.embed_batch])
        X.append(extractor.embed_batch(batch))

    X = np.concatenate(X, axis=0)
    seg_idx = np.array(seg_idx)

    # ensemble
    num_audio = len(audio_ids)
    num_classes = len(label2idx)

    logits_sum = np.zeros((num_audio, num_classes))

    X_t = torch.from_numpy(X).to(device)

    for m in models:
        with torch.no_grad():
            logits = m(X_t).cpu().numpy()

        for i, aidx in enumerate(seg_idx):
            logits_sum[aidx] += logits[i]

    logits_avg = logits_sum / (len(models) * args.tta_crops)

    pred = logits_avg.argmax(axis=1)
    pred_label = [idx2label[i] for i in pred]

    pd.DataFrame({"ID": audio_ids, "label": pred_label}).to_csv(args.out_csv, index=False)

    print(f"[Done] {args.out_csv}")

if __name__ == "__main__":
    main()