#!/bin/bash

python train.py \
  --model depth_fusion_base \
  --dataset kitti_completion \
  --data_path data \
  --crop_size 240 1216 \
  --num_workers 4 \
  --seed 123 \
  --optimizer adam \
  --weight_decay 0.0 \
  --lr 1e-4 \
  --batch_size 16 \
  --epochs 100 \
  --ckpt_dir ./checkpoints \
  --save_ckpt_freq 10 \
  --val_freq 5 \
  "$@"

