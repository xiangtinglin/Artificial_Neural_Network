#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# ============================================================
# Global paths
# ============================================================
BASE_DIR = "/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN"

METHOD1_DIR = os.path.join(BASE_DIR, "myMethods/1")
METHOD2_DIR = os.path.join(BASE_DIR, "myMethods/2_PANNs")
METHOD3_DIR = os.path.join(BASE_DIR, "myMethods/3_BEATs")

CSV_PATH = os.path.join(BASE_DIR, "gtzan.csv")
AUDIO_ROOT = os.path.join(BASE_DIR, "genres")

# 你自己的 pretrained 路徑
PANNS_CKPT = os.path.join(METHOD2_DIR, "Cnn14_mAP=0.431.pth")
BEATS_REPO_DIR = os.path.join(METHOD3_DIR, "beats")
BEATS_CKPT = os.path.join(METHOD3_DIR, "pretrained", "BEATs_iter3_plus_AS2M.pt")

# ===== 全部 ablation 統一放這裡 =====
ABLATION_ROOT = os.path.join(BASE_DIR, "myMethods", "ablation")


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str], cwd: str = None):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y))).astype(np.float32)
    return y[:target_len].astype(np.float32)


def crop_starts_uniform(n: int, crop_len: int, n_crops: int) -> np.ndarray:
    if n <= crop_len:
        return np.zeros(n_crops, dtype=np.int64)
    max_start = n - crop_len
    if n_crops == 1:
        return np.array([max_start // 2], dtype=np.int64)
    return np.round(np.linspace(0, max_start, n_crops)).astype(np.int64)


def resolve_audio_path(audio_root: str, track_id: str) -> str:
    track_id = str(track_id).strip()

    p1 = os.path.join(audio_root, track_id)
    if os.path.isfile(p1):
        return p1

    stem, ext = os.path.splitext(track_id)
    if ext == "":
        p2 = os.path.join(audio_root, track_id + ".au")
        if os.path.isfile(p2):
            return p2

    for root, _, files in os.walk(audio_root):
        if track_id in files:
            return os.path.join(root, track_id)
        if ext == "" and (track_id + ".au") in files:
            return os.path.join(root, track_id + ".au")

    raise FileNotFoundError(f"Cannot find audio for ID={track_id}")


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


# ============================================================
# Method 1: baseline CRNN eval
# ============================================================
class MusicCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            128, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        out, _ = self.gru(x)
        out = self.bn(out[-1])
        logits = self.fc(out)
        return logits, None


def extract_features_m1(path, length=128):
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    spec = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    cont = librosa.feature.spectral_contrast(y=y, sr=sr)

    feats = np.concatenate([mfcc, d1, d2, spec, chroma, cont], axis=0)
    feats = (feats - np.mean(feats, axis=1, keepdims=True)) / (np.std(feats, axis=1, keepdims=True) + 1e-6)

    if feats.shape[1] < length:
        feats = np.pad(feats, ((0, 0), (0, length - feats.shape[1])))
    else:
        feats = feats[:, :length]
    return feats.T.astype(np.float32)


def eval_method1(ckpt_path: str, csv_path: str, audio_root: str) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    hp = ckpt.get("hp", {})
    hidden_dim = hp.get("hidden_dim", 1024)
    num_layers = hp.get("num_layers", 4)
    input_dim = hp.get("input_dim", 59)
    idx2label = {i: l for l, i in ckpt["lmap"].items()}

    model = MusicCRNN(input_dim, hidden_dim, 10, num_layers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = pd.read_csv(csv_path)
    val_df = df[df["set"].str.lower() == "val"].copy()

    y_true, y_pred = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            path = resolve_audio_path(audio_root, str(row["ID"]))
            x = extract_features_m1(path)
            x = torch.from_numpy(x).unsqueeze(0).permute(1, 0, 2).to(device)
            logits, _ = model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            y_true.append(str(row["label"]))
            y_pred.append(idx2label[pred_idx])

    labels = sorted(df[df["set"].str.lower() == "train"]["label"].astype(str).unique().tolist())
    return compute_metrics(y_true, y_pred, labels)


# ============================================================
# Method 2: PANNs eval
# ============================================================
class PANNSEmbeddingExtractor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from panns_inference import AudioTagging
        self.at = AudioTagging(checkpoint_path=checkpoint_path, device=device)
        self.device = device

    @torch.no_grad()
    def embed_batch(self, wave_batch: np.ndarray) -> np.ndarray:
        _, emb = self.at.inference(wave_batch)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        return emb.astype(np.float32)


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

    def forward(self, x):
        return self.net(x)


def eval_method2(ckpt_path: str, csv_path: str, audio_root: str, tta_crops: int) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    hp = ckpt["hparams"]
    model = EmbedMLP(
        in_dim=int(hp["embed_dim"]),
        hidden_dim=int(hp["clf_hidden"]),
        num_classes=int(hp["num_classes"]),
        dropout=float(hp["clf_dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    extractor = PANNSEmbeddingExtractor(
        checkpoint_path=str(hp["panns_checkpoint"]),
        device=device,
    )

    sr = int(hp["sample_rate"])
    crop_seconds = float(hp["crop_seconds"])
    crop_len = int(round(sr * crop_seconds))

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["set"] = df["set"].astype(str).str.strip().str.lower()
    val_df = df[df["set"] == "val"].copy().reset_index(drop=True)

    audio_ids = val_df["ID"].astype(str).tolist()
    y_true = val_df["label"].astype(str).tolist()

    segments = []
    seg_audio_idx = []

    for aidx, aid in enumerate(audio_ids):
        path = resolve_audio_path(audio_root, aid)
        y = load_audio(path, sr=sr)
        starts = crop_starts_uniform(len(y), crop_len, tta_crops)
        for st in starts:
            seg = pad_or_trim(y[st:st + crop_len], crop_len)
            segments.append(seg)
            seg_audio_idx.append(aidx)

    X_list = []
    embed_batch = 32
    for i in range(0, len(segments), embed_batch):
        batch = np.stack(segments[i:i + embed_batch], axis=0).astype(np.float32)
        emb = extractor.embed_batch(batch)
        X_list.append(emb)
    X = np.concatenate(X_list, axis=0)

    logits_sum = np.zeros((len(audio_ids), int(hp["num_classes"])), dtype=np.float64)
    seg_audio_idx = np.asarray(seg_audio_idx, dtype=np.int64)

    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        logits_seg = model(X_t).detach().cpu().numpy()

    for i, aidx in enumerate(seg_audio_idx):
        logits_sum[aidx] += logits_seg[i]

    logits_avg = logits_sum / tta_crops
    pred_idx = logits_avg.argmax(axis=1)
    idx2label = ckpt["idx2label"]
    y_pred = [idx2label[int(i)] for i in pred_idx]

    labels = sorted(df[df["set"] == "train"]["label"].astype(str).unique().tolist())
    return compute_metrics(y_true, y_pred, labels)


# ============================================================
# Method 3: BEATs eval
# ============================================================
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


def eval_method3(ckpt_path: str, csv_path: str, audio_root: str, tta_crops: int) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hparams"]

    model = MLP(
        in_dim=hp["embed_dim"],
        hidden=hp["clf_hidden"],
        num_classes=hp["num_classes"],
        dropout=hp["clf_dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    extractor = BEATSEmbeddingExtractor(
        ckpt_path=hp["beats_checkpoint"],
        repo_dir=hp["beats_repo_dir"],
        device=device,
    )

    sr = int(hp["sample_rate"])
    crop_len = int(sr * hp["crop_seconds"])

    df = pd.read_csv(csv_path)
    df["set"] = df["set"].astype(str).str.lower()
    val_df = df[df["set"] == "val"].copy().reset_index(drop=True)

    audio_ids = val_df["ID"].astype(str).tolist()
    y_true = val_df["label"].astype(str).tolist()

    segments = []
    seg_audio_idx = []

    for i, aid in enumerate(audio_ids):
        path = resolve_audio_path(audio_root, aid)
        y = load_audio(path, sr)
        starts = crop_starts_uniform(len(y), crop_len, tta_crops)

        for s in starts:
            seg = pad_or_trim(y[s:s + crop_len], crop_len)
            segments.append(seg)
            seg_audio_idx.append(i)

    X = []
    embed_batch = 32
    for i in range(0, len(segments), embed_batch):
        batch = np.stack(segments[i:i + embed_batch])
        X.append(extractor.embed_batch(batch))

    X = np.concatenate(X, axis=0)
    seg_audio_idx = np.array(seg_audio_idx)

    logits_sum = np.zeros((len(audio_ids), len(ckpt["label2idx"])))
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        logits = model(X_t).cpu().numpy()

    for i, aidx in enumerate(seg_audio_idx):
        logits_sum[aidx] += logits[i]

    logits_avg = logits_sum / tta_crops
    pred = logits_avg.argmax(axis=1)
    idx2label = ckpt["idx2label"]
    y_pred = [idx2label[int(i)] for i in pred]

    labels = sorted(df[df["set"] == "train"]["label"].astype(str).unique().tolist())
    return compute_metrics(y_true, y_pred, labels)


# ============================================================
# Experiment config
# ============================================================
@dataclass
class Exp:
    name: str
    method: str
    train_args: List[str]
    ckpt_path: str
    eval_tta_crops: int


def build_experiments() -> List[Exp]:
    exps = []

    def out_dir(name: str) -> str:
        return os.path.join(ABLATION_ROOT, name)

    # ---------------- method 1 ----------------
    exps.append(
        Exp(
            name="method1_baseline",
            method="method1",
            train_args=[
                "python", os.path.join(METHOD1_DIR, "train.py"),
                "--csv", CSV_PATH,
                "--root", AUDIO_ROOT,
                "--out", out_dir("method1_baseline"),
                "--hidden_dim", "512",
                "--num_layers", "3",
                "--batch_size", "64",
                "--lr", "3e-4",
                "--weight_decay", "1e-4",
                "--epochs", "150",
                "--num_workers", "4",
            ],
            ckpt_path=os.path.join(out_dir("method1_baseline"), "best.pt"),
            eval_tta_crops=1,
        )
    )

    # ---------------- method 2 ----------------
    exps.append(
        Exp(
            name="method2_full_panns",
            method="method2",
            train_args=[
                "python", os.path.join(METHOD2_DIR, "train.py"),
                "--csv_path", CSV_PATH,
                "--audio_root", AUDIO_ROOT,
                "--out_dir", out_dir("method2_full_panns"),
                "--backend", "panns_embed",
                "--panns_checkpoint", PANNS_CKPT,
                "--sample_rate", "32000",
                "--crop_seconds", "10.0",
                "--train_crops", "12",
                "--val_crops", "9",
                "--batch_size", "256",
                "--epochs", "80",
                "--lr", "2e-4",
                "--weight_decay", "1e-2",
                "--label_smoothing", "0.1",
                "--mixup_alpha", "0.4",
                "--clf_hidden", "512",
                "--clf_dropout", "0.4",
                "--seed", "42",
            ],
            ckpt_path=os.path.join(out_dir("method2_full_panns"), "checkpoint_best.pt"),
            eval_tta_crops=21,
        )
    )

    exps.append(
        Exp(
            name="method2_wo_mixup",
            method="method2",
            train_args=[
                "python", os.path.join(METHOD2_DIR, "train.py"),
                "--csv_path", CSV_PATH,
                "--audio_root", AUDIO_ROOT,
                "--out_dir", out_dir("method2_wo_mixup"),
                "--backend", "panns_embed",
                "--panns_checkpoint", PANNS_CKPT,
                "--sample_rate", "32000",
                "--crop_seconds", "10.0",
                "--train_crops", "12",
                "--val_crops", "9",
                "--batch_size", "256",
                "--epochs", "80",
                "--lr", "2e-4",
                "--weight_decay", "1e-2",
                "--label_smoothing", "0.1",
                "--mixup_alpha", "0.0",
                "--clf_hidden", "512",
                "--clf_dropout", "0.4",
                "--seed", "42",
            ],
            ckpt_path=os.path.join(out_dir("method2_wo_mixup"), "checkpoint_best.pt"),
            eval_tta_crops=21,
        )
    )

    exps.append(
        Exp(
            name="method2_wo_label_smoothing",
            method="method2",
            train_args=[
                "python", os.path.join(METHOD2_DIR, "train.py"),
                "--csv_path", CSV_PATH,
                "--audio_root", AUDIO_ROOT,
                "--out_dir", out_dir("method2_wo_label_smoothing"),
                "--backend", "panns_embed",
                "--panns_checkpoint", PANNS_CKPT,
                "--sample_rate", "32000",
                "--crop_seconds", "10.0",
                "--train_crops", "12",
                "--val_crops", "9",
                "--batch_size", "256",
                "--epochs", "80",
                "--lr", "2e-4",
                "--weight_decay", "1e-2",
                "--label_smoothing", "0.0",
                "--mixup_alpha", "0.4",
                "--clf_hidden", "512",
                "--clf_dropout", "0.4",
                "--seed", "42",
            ],
            ckpt_path=os.path.join(out_dir("method2_wo_label_smoothing"), "checkpoint_best.pt"),
            eval_tta_crops=21,
        )
    )

    exps.append(
        Exp(
            name="method2_wo_multicrop",
            method="method2",
            train_args=[
                "python", os.path.join(METHOD2_DIR, "train.py"),
                "--csv_path", CSV_PATH,
                "--audio_root", AUDIO_ROOT,
                "--out_dir", out_dir("method2_wo_multicrop"),
                "--backend", "panns_embed",
                "--panns_checkpoint", PANNS_CKPT,
                "--sample_rate", "32000",
                "--crop_seconds", "10.0",
                "--train_crops", "1",
                "--val_crops", "1",
                "--batch_size", "256",
                "--epochs", "80",
                "--lr", "2e-4",
                "--weight_decay", "1e-2",
                "--label_smoothing", "0.1",
                "--mixup_alpha", "0.4",
                "--clf_hidden", "512",
                "--clf_dropout", "0.4",
                "--seed", "42",
            ],
            ckpt_path=os.path.join(out_dir("method2_wo_multicrop"), "checkpoint_best.pt"),
            eval_tta_crops=1,
        )
    )

    exps.append(
        Exp(
            name="method2_wo_tta",
            method="method2",
            train_args=[
                "python", os.path.join(METHOD2_DIR, "train.py"),
                "--csv_path", CSV_PATH,
                "--audio_root", AUDIO_ROOT,
                "--out_dir", out_dir("method2_wo_tta"),
                "--backend", "panns_embed",
                "--panns_checkpoint", PANNS_CKPT,
                "--sample_rate", "32000",
                "--crop_seconds", "10.0",
                "--train_crops", "12",
                "--val_crops", "9",
                "--batch_size", "256",
                "--epochs", "80",
                "--lr", "2e-4",
                "--weight_decay", "1e-2",
                "--label_smoothing", "0.1",
                "--mixup_alpha", "0.4",
                "--clf_hidden", "512",
                "--clf_dropout", "0.4",
                "--seed", "42",
            ],
            ckpt_path=os.path.join(out_dir("method2_wo_tta"), "checkpoint_best.pt"),
            eval_tta_crops=1,
        )
    )

    # ---------------- method 3 ----------------
    common_m3 = [
        "--csv_path", CSV_PATH,
        "--audio_root", AUDIO_ROOT,
        "--backend", "beats_embed",
        "--beats_checkpoint", BEATS_CKPT,
        "--beats_repo_dir", BEATS_REPO_DIR,
        "--sample_rate", "16000",
        "--crop_seconds", "10.0",
        "--batch_size", "128",
        "--epochs", "80",
        "--lr", "2e-4",
        "--weight_decay", "1e-2",
        "--clf_hidden", "1024",
        "--clf_dropout", "0.2",
        "--seed", "42",
    ]

    exps.append(
        Exp(
            name="method3_full_beats",
            method="method3",
            train_args=[
                "python", os.path.join(METHOD3_DIR, "train.py"),
                "--out_dir", out_dir("method3_full_beats"),
                *common_m3,
                "--train_crops", "12",
                "--val_crops", "9",
                "--label_smoothing", "0.05",
                "--mixup_alpha", "0.1",
            ],
            ckpt_path=os.path.join(out_dir("method3_full_beats"), "checkpoint_best.pt"),
            eval_tta_crops=21,
        )
    )

    exps.append(
        Exp(
            name="method3_wo_mixup",
            method="method3",
            train_args=[
                "python", os.path.join(METHOD3_DIR, "train.py"),
                "--out_dir", out_dir("method3_wo_mixup"),
                *common_m3,
                "--train_crops", "12",
                "--val_crops", "9",
                "--label_smoothing", "0.05",
                "--mixup_alpha", "0.0",
            ],
            ckpt_path=os.path.join(out_dir("method3_wo_mixup"), "checkpoint_best.pt"),
            eval_tta_crops=21,
        )
    )

    exps.append(
        Exp(
            name="method3_wo_label_smoothing",
            method="method3",
            train_args=[
                "python", os.path.join(METHOD3_DIR, "train.py"),
                "--out_dir", out_dir("method3_wo_label_smoothing"),
                *common_m3,
                "--train_crops", "12",
                "--val_crops", "9",
                "--label_smoothing", "0.0",
                "--mixup_alpha", "0.1",
            ],
            ckpt_path=os.path.join(out_dir("method3_wo_label_smoothing"), "checkpoint_best.pt"),
            eval_tta_crops=21,
        )
    )

    exps.append(
        Exp(
            name="method3_wo_multicrop",
            method="method3",
            train_args=[
                "python", os.path.join(METHOD3_DIR, "train.py"),
                "--out_dir", out_dir("method3_wo_multicrop"),
                *common_m3,
                "--train_crops", "1",
                "--val_crops", "1",
                "--label_smoothing", "0.05",
                "--mixup_alpha", "0.1",
            ],
            ckpt_path=os.path.join(out_dir("method3_wo_multicrop"), "checkpoint_best.pt"),
            eval_tta_crops=1,
        )
    )

    exps.append(
        Exp(
            name="method3_wo_tta",
            method="method3",
            train_args=[
                "python", os.path.join(METHOD3_DIR, "train.py"),
                "--out_dir", out_dir("method3_wo_tta"),
                *common_m3,
                "--train_crops", "12",
                "--val_crops", "9",
                "--label_smoothing", "0.05",
                "--mixup_alpha", "0.1",
            ],
            ckpt_path=os.path.join(out_dir("method3_wo_tta"), "checkpoint_best.pt"),
            eval_tta_crops=1,
        )
    )

    return exps


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="comma-separated experiment names")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    ensure_dir(ABLATION_ROOT)
    experiments = build_experiments()

    if args.only.strip():
        selected = set(x.strip() for x in args.only.split(",") if x.strip())
        experiments = [e for e in experiments if e.name in selected]

    all_rows = []

    for exp in experiments:
        print("\n" + "=" * 90)
        print(f"Experiment: {exp.name}")
        print("=" * 90)

        exp_dir = os.path.dirname(exp.ckpt_path)
        ensure_dir(exp_dir)

        metrics_path = os.path.join(exp_dir, "val_metrics.json")

        if args.skip_done and os.path.isfile(metrics_path):
            print(f"[SKIP DONE] {exp.name}")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            if not args.skip_train:
                run_cmd(exp.train_args)

            if exp.method == "method1":
                metrics = eval_method1(exp.ckpt_path, CSV_PATH, AUDIO_ROOT)
            elif exp.method == "method2":
                metrics = eval_method2(exp.ckpt_path, CSV_PATH, AUDIO_ROOT, exp.eval_tta_crops)
            elif exp.method == "method3":
                metrics = eval_method3(exp.ckpt_path, CSV_PATH, AUDIO_ROOT, exp.eval_tta_crops)
            else:
                raise ValueError(exp.method)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        row = {
            "name": exp.name,
            "method": exp.method,
            "tta_crops_eval": exp.eval_tta_crops,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
        }
        all_rows.append(row)

    summary_csv = os.path.join(ABLATION_ROOT, "summary.csv")
    summary_json = os.path.join(ABLATION_ROOT, "summary.json")

    df = pd.DataFrame(all_rows).sort_values(by=["method", "accuracy"], ascending=[True, False])
    df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print("summary_csv :", summary_csv)
    print("summary_json:", summary_json)
    print(df)


if __name__ == "__main__":
    main()