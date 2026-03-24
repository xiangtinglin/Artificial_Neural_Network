#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

Timestamp=$(date +%Y%m%d_%H%M%S)

CSV=/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/gtzan.csv
AUDIO_ROOT=/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/genres

python predict.py \
  --csv_path "${CSV}" \
  --audio_root "${AUDIO_ROOT}" \
  --checkpoint_path "/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/myMethods/3_BEATs/checkpoints_beats_seed42/checkpoint_best.pt" \
  --tta_crops 21 \
  --out_csv BEATS_submission_${Timestamp}.csv