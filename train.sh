cmd="python3 train.py \
  --batch_size 128 \
  --accumulation 512 \
  --epochs 32 \
  --emb_size 512 \
  --num_workers 1 \
  --train_dir ./data/CASIA/train \
  --test_dir ./data/CASIA/test \
  --checkpoint_path checkpoints/ \
  --random_state 42 \
  --lr 1e-1 \
  --s 30 \
  --m 0.5 \
  --reduction_factor 0.1 \
  --reduction_epochs 20 28"

echo $cmd
$cmd