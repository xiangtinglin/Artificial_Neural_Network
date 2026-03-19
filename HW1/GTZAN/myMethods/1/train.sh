#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN"
OUT_DIR="${BASE_DIR}/myMethods/1/checkpoints"

mkdir -p ${OUT_DIR}

echo "Starting Training..."

python train.py \
  --csv ${BASE_DIR}/gtzan.csv \
  --root ${BASE_DIR}/genres \
  --out ${OUT_DIR} \
  --hidden_dim 512 \
  --num_layers 3 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --epochs 150 \
  --num_workers 4