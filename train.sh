cmd="python3 train.py \
  --model irse50 \
  --batch_size 4 \
  --accumulation 4 \
  --epochs 30 \
  --emb_size 512 \
  --num_workers 0 \
  --train_dir './data/toy_casia/train' \
  --test_dir './data/toy_casia/test' \
  --eval_dir './data/LFW_mtcnn' \
  --checkpoint_path 'checkpoints/' \
  --random_state 42 \
  --lr 5e-4 \
  --s 32.0 \
  --m 0.5 \
  --warmup_epochs 4 \
  --warmup_lr 1e-6 \
  --restore_path './irse50_pretrained.pt'"

echo $cmd
$cmd