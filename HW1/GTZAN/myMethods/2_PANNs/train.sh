#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# 1) Download PANNs pretrained checkpoint (once)
# Reference: PANNs official repo (audioset_tagging_cnn) README
PANN_CKPT="Cnn14_mAP=0.431.pth"
if [ ! -f "${PANN_CKPT}" ]; then
  wget -O "${PANN_CKPT}" "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
fi

# 2) Train (seed 42)
python train.py \
  --csv_path   /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/gtzan.csv \
  --audio_root /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/genres \
  --out_dir    /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/myMethods/2_PANNs/checkpoints_panns_seed42 \
  --backend panns_embed \
  --panns_checkpoint "${PANN_CKPT}" \
  --sample_rate 32000 \
  --crop_seconds 10.0 \
  --train_crops 12 \
  --val_crops 9 \
  --batch_size 256 \
  --epochs 80 \
  --lr 2e-4 \
  --weight_decay 1e-2 \
  --label_smoothing 0.1 \
  --mixup_alpha 0.4 \
  --clf_hidden 512 \
  --clf_dropout 0.4 \
  --seed 42
