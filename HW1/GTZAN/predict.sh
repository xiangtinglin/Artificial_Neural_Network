timestamp=$(date +%Y%m%d%H%M%S)

python predict.py \
  --csv_path /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/gtzan.csv \
  --audio_root /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/genres \
  --ckpt_path /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/checkpoints_train/checkpoint_best.pt \
  --out_csv /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/checkpoints_train/61347114S_submission_${timestamp}.csv \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.1 \
  --timeseries_length 128 \
  --hop_length 512 \
  --target_sr 0
