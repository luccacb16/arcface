cmd="python3 train.py \
  --batch_size 128 \
  --accumulation 512 \
  --epochs 32 \
  --emb_size 512 \
  --num_workers 1 \
  --train_df ./data/CASIA/train.csv \
  --test_df ./data/CASIA/test.csv \
  --images_path ./data/CASIA/casia-faces \
  --checkpoint_path checkpoints/ \
  --random_state 42 \
  --lr 1e-1 \
  --s 30 \
  --m 0.5 \
  --reduction_factor 0.1 \
  --reduction_epochs 20 28"

echo $cmd
$cmd