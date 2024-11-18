import torch
import torch.nn as nn
import torch.optim as optim
#import torch.backends.cudnn as cudnn

import os
import argparse
import numpy as np
import wandb

from tqdm import tqdm
from loguru import logger
from datasets import __datasets__, get_dataloader
from models import __models__
from utils import get_lr_scheduler, get_optimizer, save_checkpoint, load_checkpoint
from utils import MetricEvaluator


parser = argparse.ArgumentParser(description="Stereo-Lidar Fusion for Depth Completion")
parser.add_argument('--model', default='depth_fusion_base', help='Select a model', choices=__models__.keys())

### Dataset
parser.add_argument('--dataset', default='kitti_completion', required=True, help='Depth completion dataset name', choices=__datasets__.keys())
parser.add_argument('--data_path', required=True, help='Dataset path')
#parser.add_argument('--training', default=True, required=True, help='Training phase or not')
parser.add_argument('--crop_size', type=tuple,default=(240, 1216), required=True, help='Training crop size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

### Model config and hyperparameters
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--optimizer', type=str, default='adam', required=True, choices=['adam', 'sgd', 'adamw'], help='Optimizer to use')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs')

### Checkpoints and logs
parser.add_argument('--ckpt_dir', required=True, help='Path to save checkpoints')
parser.add_argument('--save_ckpt_freq', default=10)
parser.add_argument('--val_freq', type=int, default=5, help='Validation step frequency')
parser.add_argument('--load_ckpt', help='Load model weights from a checkpoint')
parser.add_argument('--resume', action='store_true', help='Continue training from a checkpoint')

args = parser.parse_args()

os.makedirs(args.ckpt_dir, exist_ok=True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#cudnn.benchmark = True
#cudn.deterministic = True

def main(args):

    train_dataloader, len_dataset_train = get_dataloader(args, train=True)
    val_dataloader, len_dataset_val = get_dataloader(args, train=False)

    model = __models__[args.model](in_chans=1, convnext_pretrained=True, features_only=True)
    model.cuda()

    optimizer = get_optimizer(args, model)
    logger.log(f"Using {args.optimizer} optimizer to train the {args.model} model.")
    lr_scheduler = get_lr_scheduler(args, optimizer)
    logger.log(f"Using {args.scheduler} learning rate scheduler for the {args.optimizer} optimizer.")

    criterion_smooth_l1 = nn.SmoothL1Loss()
    criterion_l2 = nn.MSELoss(reduction='mean')

    metrics = ["mae_metric", "imae_metric", "rmse_metric", "irmse_metric", "d1_metric"]
    metric_evaluator = MetricEvaluator(metrics)

    #best_epoch = 0
    #global_step = 0
    #batch_idx_start = 0
    start_epoch = 0
    best_rmse = float('inf')

    if args.resume:

        ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {args.ckpt_dir}.")
        
        ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.ckpt')[0]))
        last_ckpt_path = os.path.join(args.ckpt_dir, ckpt_files[-1])

        logger.log(f"Loading the last pretrained model from {args.ckpt_dir}: {last_ckpt_path}")
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, last_ckpt_path) 
        logger.success(f"Successfully loaded the pretrained model from {args.ckpt_dir}")

        #batch_idx_start = global_step % len_dataset_train

    elif args.load_ckpt:
        logger.log(f"Loading the last pretrained model from {args.ckpt_dir}: {last_ckpt_path}")
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, last_ckpt_path)
        logger.success(f"Successfully loaded the pretrained model from {args.ckpt_dir}")
    
    logger.log(f"Starting training for {args.model} model on {args.dataset} dataset.")
    logger.log(f"Starting epoch {start_epoch}")

    for epoch_idx in range(start_epoch, args.epochs):

        model.train()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch_idx + 1}")
        epoch_loss = 0.0 

        for _, batch_sample in progress_bar:

            stereo, sparse, groundtruth = [x.cuda() for x in batch_sample]
            optimizer.zero_grad()
            pred = model(stereo, sparse)
            mask = (groundtruth > 0)

            #calculate loss with disparity
            loss = criterion_l2(pred[mask], groundtruth[mask]) + criterion_smooth_l1(pred[mask], groundtruth[mask])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if isinstance(lr_scheduler, optim.lr_scheduler.CyclicLR):
                lr_scheduler.step()

        if isinstance(lr_scheduler, optim.lr_scheduler.CosineAnnealingLR):
            lr_scheduler.step()

        print(f"Epoch {epoch_idx+1} Loss: {epoch_loss / len(train_dataloader)}")

        if (epoch_idx + 1) % args.val_freq == 0:
            metrics_eval = validate(model, val_dataloader, metric_evaluator)
            current_rmse = metrics_eval.get('rmse_metric', float('inf'))

            if current_rmse < best_rmse:
                logger.log(f"New best RMSE: {current_rmse:.4f} (previous: {best_rmse:.4f}) - Saving checkpoint...")
                best_rmse = current_rmse
                save_checkpoint(args.ckpt_dir, model, optimizer, lr_scheduler, epoch_idx)


            print("\nValidation Metrics:")
            for metric, value in metrics_eval.items():
                print(f"{metric:<15}: {value:.4f}")


def validate(model, val_loader, metric_evaluator):

    model.eval()
    total_metrics = {metric: 0.0 for metric in metric_evaluator.metrics}
    num_samples = 0

    progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"Validation")
    
    with torch.no_grad():
        for _, batch_sample in progress_bar:

            stereo, sparse, groundtruth = [x.cuda() for x in batch_sample]
            pred = model(stereo, sparse)

            metric_results = metric_evaluator.evaluate_metrics(pred, groundtruth)

            batch_size = groundtruth.size(0)
            num_samples += batch_size

            for metric, value in metric_results.items():
                total_metrics[metric] += value.item() * batch_size

            avg_metrics = {metric: total / num_samples for metric, total in total_metrics.items()}

            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

    final_metrics = {metric: total / num_samples for metric, total in total_metrics.items()}

    return final_metrics




