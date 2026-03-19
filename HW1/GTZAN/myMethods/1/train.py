import os
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# ============================================================
# 1) 模型定義：CNN + Bi-GRU
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
        # 由於雙向，最終維度會是 hidden_dim * 2
        self.gru = nn.GRU(128, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: (seq_len, batch, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(1, 2, 0)
        x = self.cnn(x)
        # x: (batch, 128, new_seq_len) -> (new_seq_len, batch, 128)
        x = x.permute(2, 0, 1)
        out, _ = self.gru(x)
        # 取最後一個時間步輸出
        out = self.bn(out[-1])
        logits = self.fc(out)
        return logits, None

# ============================================================
# 2) 特徵提取 (59維)
# ============================================================
def extract_features(path, length=128):
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

class AudioDataset(Dataset):
    def __init__(self, df, root, label2idx):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.label2idx = label2idx

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = extract_features(os.path.join(self.root, str(row["ID"])))
        if self.label2idx is None: return str(row["ID"]), torch.from_numpy(x)
        y = torch.tensor(self.label2idx[str(row["label"])], dtype=torch.long)
        return str(row["ID"]), torch.from_numpy(x), y

def collate(batch):
    ids = [b[0] for b in batch]
    xs = torch.stack([b[1] for b in batch], dim=0).permute(1, 0, 2)
    if len(batch[0]) == 2: return ids, xs
    ys = torch.stack([b[2] for b in batch], dim=0)
    return ids, xs, ys

def main():
    # --- 這裡補齊了所有腳本需要的參數 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default="ckpt_best")
    ap.add_argument("--input_dim", type=int, default=59)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    train_df, val_df = df[df["set"]=="train"], df[df["set"]=="val"]
    label2idx = {l: i for i, l in enumerate(sorted(train_df["label"].unique()))}
    
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'collate_fn': collate}
    train_loader = DataLoader(AudioDataset(train_df, args.root, label2idx), shuffle=True, **loader_args)
    val_loader = DataLoader(AudioDataset(val_df, args.root, label2idx), shuffle=False, **loader_args)

    model = MusicCRNN(args.input_dim, args.hidden_dim, 10, args.num_layers, args.dropout).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        acc = 0
        with torch.no_grad():
            for _, x, y in val_loader:
                logits, _ = model(x.cuda())
                acc += (torch.argmax(logits, 1) == y.cuda()).float().mean().item()
        acc /= len(val_loader)
        print(f"Epoch {epoch}/{args.epochs} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "lmap": label2idx, "hp": vars(args)}, os.path.join(args.out, "best.pt"))
            print(f"  [BEST] Saved to {args.out}/best.pt")

if __name__ == "__main__": main()