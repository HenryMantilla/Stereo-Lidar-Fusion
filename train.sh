#!/bin/bash

DATASET='kitti'
DATA_PATH='Data'
TRAINING=True
CROP_SIZE=(240,1216)
EPOCHS=50
CKPT_DIR='logs'

python train.py --dataset "$DATASET" --data_path="$DATA_PATH" --training="$TRAINING" \
    --crop_size "$CROP_SIZE" --epochs "$EPOCHS" --ckpt_dir "$CKPT_DIR"