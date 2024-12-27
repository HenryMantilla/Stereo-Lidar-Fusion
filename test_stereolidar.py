import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import disparity_to_depth, MetricEvaluator
from datasets import __datasets__, get_dataloader

import argparse

def validate(val_loader, metric_evaluator):

    total_metrics = {metric: 0.0 for metric in metric_evaluator.metrics}
    num_batches = 0

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")

    with torch.no_grad():
        for _, batch_sample in progress_bar:
            stereo, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]

            pred = torch.where(sparse > 0, sparse, stereo)

            depth_stereo = disparity_to_depth(stereo, width)
            depth_groundtruth = disparity_to_depth(groundtruth, width)
            depth_pred = disparity_to_depth(pred, width)

            # Evaluate metrics
            metric_results = metric_evaluator.evaluate_metrics(depth_stereo, depth_groundtruth)

            # Accumulate metrics
            for metric, value in metric_results.items():
                total_metrics[metric] += value.item()

            num_batches += 1

            # Update progress bar
            avg_metrics = {metric: total / num_batches for metric, total in total_metrics.items()}
            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

    # Final averaging across all batches
    avg_metrics = {metric: total / num_batches for metric, total in total_metrics.items()}

    print("\nValidation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric:<15}: {value:.4f}")
    return avg_metrics

def parse_args():

    parser = argparse.ArgumentParser(description="Stereo-Lidar Fusion for Depth Completion")
    #parser.add_argument('--model', default='depth_fusion_pvt', help='Select a model', choices=__models__.keys())

    ### Dataset
    parser.add_argument('--dataset', default='kitti_completion', help='Depth completion dataset name', choices=__datasets__.keys())
    parser.add_argument('--data_path', required=True, help='Dataset path')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 1216], help='Training crop size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    ### Model config and hyperparameters
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'cyclic'], help='Lr scheduler to use')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs')

    ### Checkpoints and logs
    parser.add_argument('--ckpt_dir', required=True, help='Path to save checkpoints')
    parser.add_argument('--save_ckpt_freq', default=10)
    parser.add_argument('--val_freq', type=int, default=5, help='Validation step frequency')
    parser.add_argument('--load_ckpt', help='Load model weights from a checkpoint')
    parser.add_argument('--resume', action='store_true', help='Continue training from a checkpoint')

    return parser.parse_args()
# Example usage
if __name__ == "__main__":

    args = parse_args()
    val_dataloader, _ = get_dataloader(args, train=False, distributed=False, rank=0, world_size=1)

    metrics = ["mae_metric", "imae_metric", "rmse_metric", "irmse_metric"]
    metric_evaluator = MetricEvaluator(metrics)
    validate(val_dataloader, metric_evaluator=metric_evaluator)
