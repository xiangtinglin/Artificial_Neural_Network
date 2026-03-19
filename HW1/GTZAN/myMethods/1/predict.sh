#!/bin/bash
# 1. 指定使用的 GPU (10GB)
export CUDA_VISIBLE_DEVICES=0

# 2. 設定路徑變數 (請確認這些路徑與你的 train.sh 一致)
BASE_DIR="/share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN"
# 指向訓練產出的最佳權重檔
CKPT_PATH="${BASE_DIR}/myMethods/1/checkpoints/best.pt"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
STUDENT_ID="61347114S"

echo "=========================================="
echo "Starting Prediction at ${TIMESTAMP}"
echo "Using Checkpoint: ${CKPT_PATH}"
echo "=========================================="

# 3. 執行預測
# 參數說明：
# --csv: 原始資料索引檔
# --root: 音訊檔案根目錄
# --ckpt: 模型權重路徑
# --out_csv: 輸出的檔名 (包含學號與時間戳記)
python predict.py \
  --csv ${BASE_DIR}/gtzan.csv \
  --root ${BASE_DIR}/genres \
  --ckpt ${CKPT_PATH} \
  --out_csv ${BASE_DIR}/myMethods/1/${STUDENT_ID}_submission_${TIMESTAMP}.csv

echo "=========================================="
echo "Done! Please upload the CSV to Kaggle."
echo "=========================================="