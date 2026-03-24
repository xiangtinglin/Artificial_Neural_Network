#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

Timestamp=$(date +%Y%m%d_%H%M%S)

CSV=/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/gtzan.csv
AUDIO_ROOT=/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/genres

# Example: single model
python predict.py \
  --csv_path "${CSV}" \
  --audio_root "${AUDIO_ROOT}" \
  --checkpoint_path "/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/myMethods/2_PANNs/checkpoints_panns_seed42/checkpoint_best.pt" \
  --tta_crops 21 \
  --out_csv 61347114S_submission_${Timestamp}.csv

# Example: ensemble of 3 seeds (best practice)
# python predict.py \
#   --csv_path "${CSV}" \
#   --audio_root "${AUDIO_ROOT}" \
#   --checkpoint_paths "/path/seed42/checkpoint_best.pt,/path/seed43/checkpoint_best.pt,/path/seed44/checkpoint_best.pt" \
#   --tta_crops 21 \
#   --out_csv submission.csv
