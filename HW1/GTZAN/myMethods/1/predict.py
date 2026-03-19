import os
import torch
import pandas as pd
import argparse
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 1) 模型定義 (必須與 train.py 完全一致)
# ============================================================
class MusicCRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(dropout)
        )
        self.gru = torch.nn.GRU(128, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.bn = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        out, _ = self.gru(x)
        out = self.bn(out[-1])
        logits = self.fc(out)
        return logits, None

# ============================================================
# 2) 特徵提取 (必須與 train.py 一致：59維)
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

class TestDataset(Dataset):
    def __init__(self, df, root):
        self.df = df.reset_index(drop=True)
        self.root = root
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = extract_features(os.path.join(self.root, str(row["ID"])))
        return str(row["ID"]), torch.from_numpy(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="submission.csv")
    args = ap.parse_args()

    # 載入 Checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # 從 ckpt 自動獲取超參數，如果沒有則使用預設
    hp = ckpt.get("hp", {})
    hidden_dim = hp.get("hidden_dim", 1024)
    num_layers = hp.get("num_layers", 4)
    input_dim = hp.get("input_dim", 59)
    idx2label = {i: l for l, i in ckpt["lmap"].items()}

    # 重建模型
    model = MusicCRNN(input_dim, hidden_dim, 10, num_layers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 讀取測試集
    df = pd.read_csv(args.csv)
    test_df = df[df["set"] == "test"].copy()
    dataset = TestDataset(test_df, args.root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    print(f"Starting inference on {len(test_df)} files...")
    with torch.no_grad():
        for audio_id, x in loader:
            # x shape: (1, 128, 59) -> 需要轉為 (128, 1, 59) 符合 GRU
            x = x.permute(1, 0, 2).to(device)
            logits, _ = model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            results.append((audio_id[0], idx2label[pred_idx]))

    # 存檔
    pd.DataFrame(results, columns=["ID", "label"]).to_csv(args.out_csv, index=False)
    print(f"Success! Prediction saved to {args.out_csv}")

if __name__ == "__main__": main()