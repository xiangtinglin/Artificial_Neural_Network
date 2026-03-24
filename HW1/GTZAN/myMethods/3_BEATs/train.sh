#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

ROOT="/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/myMethods/3_BEATs"
CSV_PATH="/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/gtzan.csv"
AUDIO_ROOT="/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/genres"

BEATS_REPO_DIR="${ROOT}/beats"
PRETRAIN_DIR="${ROOT}/pretrained"
OUT_DIR="${ROOT}/checkpoints_beats_seed42"

mkdir -p "${PRETRAIN_DIR}"
mkdir -p "${OUT_DIR}"

# --------------------------------------------------
# 1) Download BEATs source code if beats/ not exists
# --------------------------------------------------
if [ ! -f "${BEATS_REPO_DIR}/BEATs.py" ]; then
  echo "[INFO] beats/ not found. Cloning Microsoft unilm repo..."
  TMP_DIR="${ROOT}/unilm_tmp_$$"
  git clone https://github.com/microsoft/unilm.git "${TMP_DIR}"
  mv "${TMP_DIR}/beats" "${BEATS_REPO_DIR}"
  rm -rf "${TMP_DIR}"
  echo "[INFO] beats/ prepared at ${BEATS_REPO_DIR}"
fi

# --------------------------------------------------
# 2) Set checkpoint path
# --------------------------------------------------
# Please put your downloaded BEATs checkpoint here.
# Example filename:
#   BEATs_iter3_plus_AS2M.pt
BEATS_CKPT="${PRETRAIN_DIR}/BEATs_iter3_plus_AS2M.pt"

if [ ! -f "${BEATS_CKPT}" ]; then
  echo "[ERROR] BEATs checkpoint not found:"
  echo "        ${BEATS_CKPT}"
  echo
  echo "Please download a BEATs pretrained checkpoint from the official README model table"
  echo "and place it at the path above, then rerun this script."
  exit 1
fi

# --------------------------------------------------
# 3) Train
# --------------------------------------------------
python "${ROOT}/train.py" \
  --csv_path "${CSV_PATH}" \
  --audio_root "${AUDIO_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --backend beats_embed \
  --beats_checkpoint "${BEATS_CKPT}" \
  --beats_repo_dir "${BEATS_REPO_DIR}" \
  --sample_rate 16000 \
  --crop_seconds 10.0 \
  --train_crops 12 \
  --val_crops 9 \
  --batch_size 128 \
  --epochs 80 \
  --lr 2e-4 \
  --weight_decay 1e-2 \
  --label_smoothing 0.05 \
  --mixup_alpha 0.1 \
  --clf_hidden 1024 \
  --clf_dropout 0.2 \
  --seed 42