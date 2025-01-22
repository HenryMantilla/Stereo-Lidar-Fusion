import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

import os
import argparse
import numpy as np
import wandb
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
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['multi_step', 'cosine', 'cyclic'], help='Lr scheduler to use')
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
    #dense_samples = dense_lidar[:num_samples].detach().cpu()
    #depth_convnext_samples = depth_convnext[:num_samples].detach().cpu()

    gt_samples = apply_colormap(gt_samples, colormap='magma')
    stereo_samples = apply_colormap(stereo_samples, colormap='magma')
    init_samples = apply_colormap(init_samples, colormap='magma')
    final_samples = apply_colormap(final_samples, colormap='magma')
    confidence_samples = apply_colormap(confidence_samples, colormap='magma')
    #dense_samples = apply_colormap(dense_samples, colormap='magma')
    #depth_convnext_samples = apply_colormap(depth_convnext_samples, colormap='magma')

    gt_grid = make_grid(gt_samples, nrow=num_samples, normalize=False, scale_each=False)
    stereo_grid = make_grid(stereo_samples, nrow=num_samples, normalize=False, scale_each=False)
    init_grid = make_grid(init_samples, nrow=num_samples, normalize=False, scale_each=False)
    final_grid = make_grid(final_samples, nrow=num_samples, normalize=False, scale_each=False)
    confidence_grid = make_grid(confidence_samples, nrow=num_samples, normalize=False, scale_each=False)
    #dense_grid = make_grid(dense_samples, nrow=num_samples, normalize=False, scale_each=False)
    #depth_convnext_grid = make_grid(depth_convnext_samples, nrow=num_samples, normalize=False, scale_each=False)

    wandb.log({
        "Groundtruth Depth": wandb.Image(gt_grid, caption=caption),
        "Stereo Inputs": wandb.Image(stereo_grid, caption=caption),
        "Init pred": wandb.Image(init_grid, caption=caption),
        "Final pred": wandb.Image(final_grid, caption=caption),
        "Confidence": wandb.Image(confidence_grid, caption=caption),
        #"First Refinement": wandb.Image(dense_grid, caption=caption),
        #"Depth Convnext": wandb.Image(depth_convnext_grid, caption=caption)
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


def l1l2_loss(predictions, ground_truth, gamma):
    """
    Computes the custom loss described in the equation using PyTorch's built-in loss functions.
    
    Args:
        predictions (list of torch.Tensor): List of predicted depth maps [D^2, D^3, ..., D^N].
        ground_truth (torch.Tensor): Ground truth depth map (D_gt) with shape [B, H, W].
        gamma (float): Exponential weight factor.
    
    Returns:
        torch.Tensor: The computed loss value.
    """
    # Loss functions
    l1_loss_fn = nn.L1Loss(reduction='mean')
    mse_loss_fn = nn.MSELoss(reduction='mean')
    
    # Create mask for valid ground truth values (D_gt > 0)
    mask = ground_truth > 0

    total_loss = 0.0
    N = len(predictions) + 1  # Number of predictions including D^2 ... D^N

    # Loop over predictions (D^2 to D^N)
    for i, D_hat in enumerate(predictions, start=2):
        # Apply the mask
        valid_pred = D_hat[mask]
        valid_gt = ground_truth[mask]

        # Compute weighted L1 and L2 losses
        weight = gamma ** (N - i)
        l1_loss = l1_loss_fn(valid_pred, valid_gt)
        l2_loss = mse_loss_fn(valid_pred, valid_gt)  # MSE is equivalent to L2 loss

        # Combine losses for this prediction
        total_loss += weight * (l1_loss + l2_loss)

    return total_loss


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
            config=args,
            mode='online'
            #id='vdgozqfe',
            #resume="allow"
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
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #for name, param in model.named_parameters():
        #if name.startswith('convnext_encoder'):
        #    param.requires_grad = False
        #    print(f"Froze parameter: {name}")
        #if name.startswith('refinement_stage2'):
        #    param.requires_grad = False
        #    print(f"Froze parameter: {name}")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True) #find_unused_parameters=True

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    #scaler = GradScaler()

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
        
        ckpt_files.sort(key=lambda x: int(x.split('_')[-1].split('.ckpt')[0]))
        last_ckpt_path = os.path.join(args.ckpt_dir, ckpt_files[-1])

        if rank == 0:
            logger.info(f"Loading the last pretrained model from {last_ckpt_path}")
        
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, last_ckpt_path, weights_only=False)
        
        if rank == 0:
            logger.success(f"Successfully loaded the pretrained model from {args.ckpt_dir}")
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    
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
            #with autocast(device_type='cuda', dtype=torch.float16):
            init_pred, final_pred, _, preds = model(rgb, stereo, sparse)

            if args.model == 'depth_fusion_swin':
                rgb = rgb[:, :, :256, :1216]
                #confidence = confidence[:, :, :256, :1216]
                init_pred = init_pred[:, :, :256, :1216]
                final_pred = final_pred[:, :, :256, :1216]

            mask = (groundtruth > 1e-8)

            #loss_residual = 0.4 * criterion_l2(dense_lidar[mask]+init_pred[mask], groundtruth[mask]) + criterion_l1(dense_lidar[mask]+init_pred[mask], groundtruth[mask])
            #loss = 0.7 * criterion_l2(final_pred[mask], groundtruth[mask]) + criterion_l1(final_pred[mask], groundtruth[mask])
            loss = l1l2_loss(preds, groundtruth, gamma=0.6)
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

            #if isinstance(lr_scheduler, optim.lr_scheduler.CyclicLR):
            #    lr_scheduler.step()

        #if isinstance(lr_scheduler, optim.lr_scheduler.CosineAnnealingLR):
        #    lr_scheduler.step()

        #elif isinstance(lr_scheduler, optim.lr_scheduler.MultiStepLR):
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
    total_samples = 0  

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(rank != 0))

    with torch.no_grad():
        for batch_idx, batch_sample in progress_bar:
            rgb, stereo, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]

            if args.model == 'depth_fusion_swin':
                rgb = input_padding(rgb)
                stereo = input_padding(stereo)
                sparse = input_padding(sparse)

            init_pred, final_pred, confidence, preds = model(rgb, stereo, sparse)

            if args.model == 'depth_fusion_swin':
                stereo = stereo[:, :, :256, :1216]
                init_pred = init_pred[:, :, :256, :1216]
                final_pred = final_pred[:, :, :256, :1216]
                confidence = confidence[:, :, :256, :1216]

            # Evaluate metrics for the current batch
            batch_metrics = metric_evaluator.evaluate_metrics(final_pred, groundtruth) #change in stage 2 to final_pred
            batch_size = groundtruth.size(0)  

            for metric, value in batch_metrics.items():
                total_metrics[metric] += value.item() * batch_size 

            total_samples += batch_size

            avg_metrics = {metric: total / total_samples for metric, total in total_metrics.items()}
            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

            if batch_idx == 0 and rank == 0:
                log_predictions(rank, 
                                groundtruth, 
                                stereo, 
                                init_pred, 
                                final_pred, 
                                confidence, 
                                num_samples=min(batch_size, 1), caption="Validation Predictions")

    # Distributed reduction if applicable
    if dist.is_initialized():
        for metric in total_metrics:
            metric_tensor = torch.tensor(total_metrics[metric], dtype=torch.float32, device="cuda")
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            total_metrics[metric] = metric_tensor.item()

        total_samples_tensor = torch.tensor(total_samples, dtype=torch.float32, device="cuda")
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = int(total_samples_tensor.item())

    # Final average metrics
    avg_metrics = {metric: total / total_samples for metric, total in total_metrics.items()}

    if rank == 0:
        wandb.log(avg_metrics)

    return avg_metrics


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args=args)

if __name__ == '__main__':
    main()
