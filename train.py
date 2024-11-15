import torch
import torch.nn as nn

import os
import argparse
import numpy as np

from tqdm import tqdm
from datasets import __datasets__, fetch_dataloader
from models import __models__


parser = argparse.ArgumentParser(description="Stereo-Lidar Fusion for Depth Completion")
parser.add_argument('--model', default='depth_fusion_base', help='Select a model', choices=__models__.keys())

### Dataset
parser.add_argument('--dataset', default='kitti_completion', required=True, help='Depth completion dataset name', choices=__datasets__.keys())
parser.add_argument('--data_path', required=True, help='Dataset path')
parser.add_argument('--training', default=True, required=True, help='Training phase or not')
parser.add_argument('--crop_size', type=tuple,default=(240, 1216), required=True, help='Training crop size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

### Hyperparameters 
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
parser.add_argument('--epochs', type=int, required=True, help='Number of train epochs')

### Checkpoints and logs
parser.add_argument('--ckpt_dir', required=True, help='Path to save checkpoints')
parser.add_argument('--load_ckpt', help='Load model weights from a checkpoint')
parser.add_argument('--resume', action='store_true', help='Continue training from a checkpoint')

args = parser.parse_args()

os.makedirs(args.ckpt_dir, exist_ok=True)

### Seeds 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


### Dataloader

train_dataloader = fetch_dataloader(args)

print(train_dataloader)
