import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

import os
import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from loguru import logger
from datasets import __datasets__, get_dataloader
from models import __models__
from utils import get_lr_scheduler, get_optimizer, save_checkpoint, load_checkpoint, disparity_to_depth, MetricEvaluator

def parse_args():

    parser = argparse.ArgumentParser(description="Stereo-Lidar Fusion for Depth Completion")
    parser.add_argument('--model', default='depth_fusion_pvt', help='Select a model', choices=__models__.keys())

    ### Dataset
    parser.add_argument('--dataset', default='kitti_completion', help='Depth completion dataset name', choices=__datasets__.keys())
    parser.add_argument('--data_path', required=True, help='Dataset path')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 1242], help='Training crop size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    ### Model config and hyperparameters
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'cyclic'], help='Lr scheduler to use')
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

    return parser.parse_args()

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    return rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def hard_negative_loss(pred, target, percentage=0.1):
    diff = torch.abs(pred - target)
    num_hard_pixels = int(percentage * diff.numel())
    hard_pixels = torch.topk(diff.view(-1), num_hard_pixels).values
    return hard_pixels.mean()


def log_predictions(rank, pred, pred_refined, groundtruth, stereo, num_samples, caption="Predictions"):

    if rank != 0:
        return
    
    pred_samples = pred[:num_samples].detach().cpu()
    pred_refined_samples = pred_refined[:num_samples].detach().cpu()
    gt_samples = groundtruth[:num_samples].detach().cpu()
    stereo_samples = stereo[:num_samples].detach().cpu()

    pred_refined_samples = apply_colormap(pred_refined_samples, colormap='magma')
    pred_samples = apply_colormap(pred_samples, colormap='magma')
    gt_samples = apply_colormap(gt_samples, colormap='magma')
    stereo_samples = apply_colormap(stereo_samples, colormap='magma')

    predictions_refined_grid = make_grid(pred_refined_samples, nrow=num_samples, normalize=True, scale_each=True)
    predictions_grid = make_grid(pred_samples, nrow=num_samples, normalize=True, scale_each=True)
    gt_grid = make_grid(gt_samples, nrow=num_samples, normalize=False, scale_each=False)
    stereo_grid = make_grid(stereo_samples, nrow=num_samples, normalize=False, scale_each=False)

    wandb.log({
        "Fused Depth": wandb.Image(predictions_grid, caption=caption),
        "Refined Depth": wandb.Image(predictions_refined_grid, caption=caption),
        "Groundtruth Depth": wandb.Image(gt_grid, caption=caption),
        "Stereo Inputs": wandb.Image(stereo_grid, caption=caption),
    })

def apply_colormap(image_tensor, colormap='plasma'):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0).squeeze(0)

    image_np = image_tensor.numpy()
    normalized_image = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-6)

    colormap_fn = cm.get_cmap(colormap)
    color_image = colormap_fn(normalized_image)

    color_image = (color_image[:, :, :3] * 255).astype(np.uint8)
    color_image = torch.from_numpy(color_image).permute(2,0,1)

    return color_image.unsqueeze(0).float()

def input_padding(x, target_width=1280): #for swin transformer

    B, C, H, W = x.shape

    pad_w = target_width - W
    pad_h = 0  
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))  

    return x_padded

def freeze_refinement(fusion_model, freeze=True):
    # Handle DDP-wrapped models
    model = fusion_model.module if isinstance(fusion_model, torch.nn.parallel.DistributedDataParallel) else fusion_model

    # Modify requires_grad for the refinement module
    for name, param in model.named_parameters():
        if 'refinement' in name:
            param.requires_grad = not freeze
        else:
            param.requires_grad = True  

def train(args):

    rank, world_size = setup_distributed()
    torch.distributed.barrier()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True


    if rank == 0:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            project='Depth-Completion',
            name=f'Fusion_{args.crop_size}_{args.batch_size}_{args.lr}_{args.epochs}_{args.model}',
            config=args
        )
    else:
        os.environ["WANDB_MODE"] = "offline"

    train_dataloader, _ = get_dataloader(args, train=True, distributed=(world_size > 1), rank=rank, world_size=world_size)
    val_dataloader, _ = get_dataloader(args, train=False, distributed=(world_size > 1), rank=rank, world_size=world_size)

    if args.model == 'depth_fusion_pvt':
        model = __models__[args.model](convnext_pretrained=True)
    elif args.model == 'depth_fusion_swin':
        model = __models__[args.model](convnext_pretrained=True)
    elif args.model == 'depth_fusion_icra':
        model = __models__[args.model](in_channels=1)

    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    criterion_l1 = nn.L1Loss(reduction='mean')
    criterion_l2 = nn.MSELoss(reduction='mean')

    metrics = ["mae_metric", "imae_metric", "rmse_metric", "irmse_metric"]
    metric_evaluator = MetricEvaluator(metrics)

    start_epoch = 0
    best_rmse = float('inf')


    if args.resume or args.load_ckpt:
        ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {args.ckpt_dir}.")
        
        ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.ckpt')[0]))
        last_ckpt_path = os.path.join(args.ckpt_dir, ckpt_files[-1])

        if rank == 0:
            logger.info(f"Loading the last pretrained model from {args.ckpt_dir}: {last_ckpt_path}")
        
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, last_ckpt_path, weights_only=False)
        
        if rank == 0:
            logger.success(f"Successfully loaded the pretrained model from {args.ckpt_dir}")
        torch.distributed.barrier()
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_millions = total_params / 1e6
        logger.info(f"Model has {total_params_millions:.2f} million parameters.")
        logger.info(f"Using {args.optimizer} optimizer to train the {args.model} model.")
        logger.info(f"Using {args.scheduler} learning rate scheduler for the {args.optimizer} optimizer.")
        logger.info(f"Starting training for {args.model} model on {args.dataset} dataset.")
        logger.info(f"Starting epoch {start_epoch}")


    for epoch_idx in range(start_epoch, args.epochs):

        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        model.train()
        progress_bar = tqdm(enumerate(train_dataloader), 
                            total=len(train_dataloader), 
                            desc=f"Epoch {epoch_idx + 1}",
                            disable=(rank != 0))
        
        epoch_loss = 0.0

        for _, batch_sample in progress_bar:
            rgb, stereo, sparse, groundtruth, _ = [x.cuda(non_blocking=True) for x in batch_sample]

            optimizer.zero_grad()
            fused_disparity, refined_disparity = model(rgb, stereo, sparse) 

            mask = (groundtruth > 1e-8)

            #if epoch_idx < 50:
            #    loss_fused = criterion_l1(fused_disparity[mask], groundtruth[mask]) +  0.4 * criterion_l1(fused_disparity, stereo)
            #    loss_refined = loss_fused
            #    loss = loss_fused
            #else:
            loss_fused = criterion_l2(fused_disparity[mask], groundtruth[mask]) +  0.4 * criterion_l1(fused_disparity, stereo)
            loss_refined = criterion_l2(refined_disparity[mask], groundtruth[mask]) + 0.4 * criterion_l1(refined_disparity, stereo)
            loss = loss_fused + loss_refined

            loss.backward()
            optimizer.step()

            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss)
            reduced_loss /= world_size

            epoch_loss += reduced_loss.item()

            if rank == 0:
                progress_bar.set_postfix(loss=reduced_loss.item())

                wandb.log({
                    "batch_loss": reduced_loss.item(),
                    "fused_loss": loss_fused.item(),
                    "refined_loss": loss_refined.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

            if isinstance(lr_scheduler, optim.lr_scheduler.CyclicLR):
                lr_scheduler.step()

        if isinstance(lr_scheduler, optim.lr_scheduler.CosineAnnealingLR):
            lr_scheduler.step()

        elif isinstance(lr_scheduler, optim.lr_scheduler.StepLR):
            lr_scheduler.step()

        epoch_loss_tensor = torch.tensor(epoch_loss, device='cuda')
        dist.all_reduce(epoch_loss_tensor)
        avg_epoch_loss = epoch_loss_tensor.item() / len(train_dataloader)

        if rank == 0:
            wandb.log({"Epoch Loss": avg_epoch_loss})
            print(f"Epoch {epoch_idx+1} Loss: {avg_epoch_loss}")

        if (epoch_idx + 1) % args.val_freq == 0:
            metrics_eval = validate(rank, world_size, model, val_dataloader, metric_evaluator, args)
            dist.barrier()

            if rank == 0:
                current_rmse = metrics_eval.get('rmse_metric', float('inf'))

                if current_rmse < best_rmse:
                    logger.info(f"New best RMSE: {current_rmse:.4f} (previous: {best_rmse:.4f}) - Saving checkpoint...")
                    best_rmse = current_rmse
                    save_checkpoint(args.ckpt_dir, model, optimizer, lr_scheduler, epoch_idx)


                print("\nValidation Metrics:")
                for metric, value in metrics_eval.items():
                    print(f"{metric:<15}: {value:.4f}")

    torch.distributed.barrier()
    cleanup()

def validate(rank, world_size, model, val_loader, metric_evaluator, args):

    model.eval()
    total_metrics = {metric: 0.0 for metric in metric_evaluator.metrics}
    num_batches = 0

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(rank != 0))

    with torch.no_grad():
        for batch_idx, batch_sample in progress_bar:
            rgb, stereo, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]

            fused_disparity, refined_disparity = model(rgb, stereo, sparse)

            depth_stereo = disparity_to_depth(stereo, width)
            depth_groundtruth = disparity_to_depth(groundtruth, width)
            depth_pred = disparity_to_depth(fused_disparity, width)
            depth_pred_refined = disparity_to_depth(refined_disparity, width)

            metric_results = metric_evaluator.evaluate_metrics(depth_pred, depth_groundtruth)

            batch_size = groundtruth.size(0)
            num_batches += batch_size

            for metric, value in metric_results.items():
                total_metrics[metric] += value.item() * batch_size

            avg_metrics = {metric: total / num_batches for metric, total in total_metrics.items()}
            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

            if batch_idx == 0 and rank == 0:
                log_predictions(rank, depth_pred, depth_pred_refined, depth_groundtruth, depth_stereo, num_samples=1, caption="Validation Predictions")

    avg_metrics = {metric: total / num_batches for metric, total in total_metrics.items()}
    if dist.is_initialized():
        for metric in avg_metrics:
            metric_tensor = torch.tensor(avg_metrics[metric], dtype=torch.float32, device="cuda")
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            avg_metrics[metric] = metric_tensor.item() / dist.get_world_size()

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        wandb.log(avg_metrics)

    return avg_metrics


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args=args)



if __name__ == '__main__':
    main()