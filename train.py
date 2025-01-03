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

def log_predictions(rank, groundtruth, stereo, init_pred, final_pred, confidence, num_samples, caption="Predictions"):

    if rank != 0:
        return
    
    gt_samples = groundtruth[:num_samples].detach().cpu()
    stereo_samples = stereo[:num_samples].detach().cpu()
    init_samples = init_pred[:num_samples].detach().cpu()
    final_samples = final_pred[:num_samples].detach().cpu()
    confidence_samples = confidence[:num_samples].detach().cpu()

    gt_samples = apply_colormap(gt_samples, colormap='magma')
    stereo_samples = apply_colormap(stereo_samples, colormap='magma')
    init_samples = apply_colormap(init_samples, colormap='magma')
    final_samples = apply_colormap(final_samples, colormap='magma')
    confidence_samples = apply_colormap(confidence_samples, colormap='magma')

    gt_grid = make_grid(gt_samples, nrow=num_samples, normalize=False, scale_each=False)
    stereo_grid = make_grid(stereo_samples, nrow=num_samples, normalize=False, scale_each=False)
    init_grid = make_grid(init_samples, nrow=num_samples, normalize=True, scale_each=True)
    final_grid = make_grid(final_samples, nrow=num_samples, normalize=True, scale_each=True)
    confidence_grid = make_grid(confidence_samples, nrow=num_samples, normalize=True, scale_each=True)

    wandb.log({
        "Groundtruth Depth": wandb.Image(gt_grid, caption=caption),
        "Stereo Inputs": wandb.Image(stereo_grid, caption=caption),
        "CNN pred": wandb.Image(init_grid, caption=caption),
        "Transformer pred": wandb.Image(final_grid, caption=caption),
        "Confidence": wandb.Image(confidence_grid, caption=caption)
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

def input_padding(x, target_width=1280):

    B, C, H, W = x.shape

    pad_w = target_width - W
    pad_h = 0  
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))  

    return x_padded

def freeze_refinement(fusion_model, module, freeze=True):

    model = fusion_model.module if isinstance(fusion_model, torch.nn.parallel.DistributedDataParallel) else fusion_model

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
        if module in name:
            param.requires_grad = not freeze
        else:
            param.requires_grad = True

def update_optimizer(optimizer, model, lr):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_defaults = optimizer.defaults.copy()
    optimizer_defaults['lr'] = lr

    new_optimizer = type(optimizer)(params, **optimizer_defaults)

    return new_optimizer


def edge_aware_smoothness_per_pixel(rgb, depth):
    def gradient_y(img):
        kernel = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)  
        kernel = kernel.repeat(img.shape[1], 1, 1, 1) 
        return F.conv2d(img, kernel, padding=1)

    def gradient_x(img):
        kernel = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)  
        kernel = kernel.repeat(img.shape[1], 1, 1, 1)  
        return F.conv2d(img, kernel, padding=1)

    depth_gradients_x = gradient_x(depth)
    depth_gradients_y = gradient_y(depth)

    rgb_gradients_x = torch.mean(torch.stack([gradient_x(rgb[:, i:i+1, :, :]) for i in range(3)]), dim=0)
    rgb_gradients_y = torch.mean(torch.stack([gradient_y(rgb[:, i:i+1, :, :]) for i in range(3)]), dim=0)

    weights_x = torch.exp(-torch.abs(rgb_gradients_x))
    weights_y = torch.exp(-torch.abs(rgb_gradients_y))

    # Compute edge-aware smoothness
    smoothness_x = torch.abs(depth_gradients_x) * weights_x
    smoothness_y = torch.abs(depth_gradients_y) * weights_y

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def compute_depth_losses(predicted_depths, ground_truth_depth, args):

    l1_losses = []
    l2_losses = []
    
    for pred in predicted_depths:

        if args.model == 'depth_fusion_swin':
            pred = pred[:, :, :256, :1216]
        valid_mask = (ground_truth_depth > 1e-8)

        pred = pred[valid_mask]
        gt = ground_truth_depth[valid_mask]
        
        l1_loss = F.l1_loss(pred, gt, reduction='mean')  
        l2_loss = F.mse_loss(pred, gt, reduction='mean') 
        
        l1_losses.append(l1_loss)
        l2_losses.append(l2_loss)
    
    mean_l1_loss = torch.mean(torch.stack(l1_losses))
    mean_l2_loss = torch.mean(torch.stack(l2_losses))
    
    result = mean_l1_loss.item() + mean_l2_loss.item()
    return result


def train(args):

    rank, world_size = setup_distributed()
    #torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    #cudnn.deterministic = True

    if rank == 0:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            project='Depth-Completion',
            name=f'Fusion_{args.crop_size}_{args.batch_size}_{args.lr}_{args.epochs}',
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

    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True)

    #freeze_refinement(model, 'refinement', True)

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
        #torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    
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

        for batch_idx, batch_sample in progress_bar:
            rgb, stereo, sparse, groundtruth, _ = [x.cuda(non_blocking=True) for x in batch_sample]
            if args.model == 'depth_fusion_swin':
                rgb = input_padding(rgb)
                stereo = input_padding(stereo)
                sparse = input_padding(sparse)

            optimizer.zero_grad()
            init_pred, final_pred, confidence = model(rgb, stereo, sparse)

            if args.model == 'depth_fusion_swin':
                rgb = rgb[:, :, :256, :1216]
                confidence = confidence[:, :, :256, :1216]
                init_pred = init_pred[:, :, :256, :1216]
                final_pred = final_pred[:, :, :256, :1216]

            mask = (groundtruth > 1e-8)

            #loss_inter = compute_depth_losses(inter_upsamples, groundtruth, args)
            loss_edges = edge_aware_smoothness_per_pixel(rgb, final_pred)
            loss_init = criterion_l2(init_pred[mask], groundtruth[mask]) + 0.5 * criterion_l1(init_pred[mask], groundtruth[mask])
            loss_final = criterion_l2(final_pred[mask], groundtruth[mask]) + 0.5 * criterion_l1(final_pred[mask], groundtruth[mask])

            loss = 0.6 * loss_edges + 0.5 * loss_init + loss_final
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
                    "prediction_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                """
                if epoch_idx % 5 == 0 and batch_idx == 1:
                    wandb.log({
                        "Pred training": wandb.Image(final_pred, caption="Training"),
                        "RGB training": wandb.Image(rgb, caption="Training"),
                        "Stereo training": wandb.Image(stereo, caption="Training"),
                        "Lidar training": wandb.Image(sparse, caption="Training"),
                        "GT Training": wandb.Image(groundtruth, caption="Training")
                    })
                """
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
            metrics_eval = validate(rank, model, val_dataloader, metric_evaluator, args)
            dist.barrier(device_ids=[torch.cuda.current_device()])

            if rank == 0:
                current_rmse = metrics_eval.get('rmse_metric', float('inf'))

                if current_rmse < best_rmse:
                    logger.info(f"New best RMSE: {current_rmse:.4f} (previous: {best_rmse:.4f}) - Saving checkpoint...")
                    best_rmse = current_rmse
                    save_checkpoint(args.ckpt_dir, model, optimizer, lr_scheduler, epoch_idx)


                print("\nValidation Metrics:")
                for metric, value in metrics_eval.items():
                    print(f"{metric:<15}: {value:.4f}")

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    cleanup()

def validate(rank, model, val_loader, metric_evaluator, args):

    model.eval()
    total_metrics = {metric: 0.0 for metric in metric_evaluator.metrics}
    num_batches = 0

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(rank != 0))

    with torch.no_grad():
        for batch_idx, batch_sample in progress_bar:
            rgb, stereo, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]

            if args.model == 'depth_fusion_swin':
                rgb = input_padding(rgb)
                stereo = input_padding(stereo)
                sparse = input_padding(sparse)

            init_pred, final_pred, confidence = model(rgb, stereo, sparse)

            if args.model == 'depth_fusion_swin':
                stereo = stereo[:, :, :256, :1216]
                init_pred = init_pred[:, :, :256, :1216]
                final_pred = final_pred[:, :, :256, :1216]
                confidence = confidence[:, :, :256, :1216]

            #depth_stereo = disparity_to_depth(stereo, width)
            #depth_groundtruth = disparity_to_depth(groundtruth, width)
            #depth_pred = disparity_to_depth(final_pred, width)
            #depth_init = disparity_to_depth(init_pred, width)

            metric_results = metric_evaluator.evaluate_metrics(final_pred, groundtruth)

            for metric, value in metric_results.items():
                total_metrics[metric] += value.item() 

            num_batches += 1
            avg_metrics = {metric: total / num_batches for metric, total in total_metrics.items()}
            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

            if batch_idx == 0 and rank == 0:
                #log_predictions(rank, depth_groundtruth, depth_stereo, depth_init, depth_pred, confidence, num_samples=1, caption="Validation Predictions")
                log_predictions(rank, groundtruth, stereo, init_pred, final_pred, confidence, num_samples=1, caption="Validation Predictions")

    avg_metrics = {metric: total / num_batches for metric, total in total_metrics.items()}
    if dist.is_initialized():
        for metric in avg_metrics:
            metric_tensor = torch.tensor(avg_metrics[metric], dtype=torch.float32, device="cuda")
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            avg_metrics[metric] = metric_tensor.item() / dist.get_world_size()

    if rank == 0:
        wandb.log(avg_metrics)

    return avg_metrics


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args=args)



if __name__ == '__main__':
    main()
