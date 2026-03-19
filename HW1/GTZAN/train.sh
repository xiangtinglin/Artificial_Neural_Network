# pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES = 0
python train.py \
  --csv_path /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/gtzan.csv \
  --audio_root /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/genres \
  --out_dir /share/nas165/xiangtinglin/Project/Lesson/114-2/Artificial_Neural_Network/HW1/GTZAN/checkpoints_train \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --epochs 100 \
  --validate_every 10 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --timeseries_length 128 \
  --hop_length 512 \
  --target_sr 22050