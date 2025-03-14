import os
import argparse
import numpy as np
import wandb
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

from tqdm import tqdm
from loguru import logger
from datasets import __datasets__, get_dataloader
from models import __models__
from utils import get_lr_scheduler, get_optimizer, save_checkpoint, load_checkpoint, MetricEvaluator, depth_to_disparity_train

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

    # Architecture choices
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range") #768 default from IGEV++ 192 for IGEV
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")

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

def log_predictions(rank, rgb_left, rgb_right, gt_depth, gt_disp, stereo, init_pred, final_pred, res_disp, res_depth, caption="Predictions"):

    if rank != 0:
        return

    num_samples = 1
    
    rgb_left_sample = rgb_left[:num_samples].detach().cpu()
    rgb_right_sample = rgb_right[:num_samples].detach().cpu()
    gt_sample = gt_depth[:num_samples].detach().cpu()
    gt_disp_sample = gt_disp[:num_samples].detach().cpu()
    stereo_sample = stereo[:num_samples].detach().cpu()
    init_sample = init_pred[:num_samples].detach().cpu()
    final_sample = final_pred[:num_samples].detach().cpu()
    res_disp_sample = res_disp[:num_samples].detach().cpu()
    res_depth_sample = res_depth[:num_samples].detach().cpu()

    gt_sample = apply_colormap(gt_sample, colormap='magma')
    gt_disp_sample = apply_colormap(gt_disp_sample, colormap='magma')
    stereo_sample = apply_colormap(stereo_sample, colormap='magma')
    init_sample = apply_colormap(init_sample, colormap='magma')
    final_sample = apply_colormap(final_sample, colormap='magma')
    res_disp_sample = apply_colormap(res_disp_sample, colormap='magma')
    res_depth_sample = apply_colormap(res_depth_sample, colormap='magma')

    rgb_left_grid = make_grid(rgb_left_sample.unsqueeze(0))
    rgb_right_grid = make_grid(rgb_right_sample.unsqueeze(0))
    gt_grid = make_grid(gt_sample.unsqueeze(0))
    gt_disp_grid = make_grid(gt_disp_sample.unsqueeze(0))
    stereo_grid = make_grid(stereo_sample.unsqueeze(0))
    init_grid = make_grid(init_sample.unsqueeze(0))
    final_grid = make_grid(final_sample.unsqueeze(0))
    res_disp_grid = make_grid(res_disp_sample.unsqueeze(0))
    res_depth_grid = make_grid(res_depth_sample.unsqueeze(0))

    wandb.log({
        "Left Image": wandb.Image(rgb_left_grid, caption=caption),
        "Right Image": wandb.Image(rgb_right_grid, caption=caption),
        "Groundtruth Depth": wandb.Image(gt_grid, caption=caption),
        "Groundtruth Disparity": wandb.Image(gt_disp_grid, caption=caption),
        "Stereo Inputs": wandb.Image(stereo_grid, caption=caption),
        "Init pred": wandb.Image(init_grid, caption=caption),
        "Final pred": wandb.Image(final_grid, caption=caption),
        "Disparity Residual": wandb.Image(res_disp_grid, caption=caption),
        "Depth Residual": wandb.Image(res_depth_grid, caption=caption),
    })

def apply_colormap(image_tensor, colormap='plasma'):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0).squeeze(0)

    image_np = image_tensor.float().cpu().numpy()
    normalized_image = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-6)

    colormap_fn = cm.get_cmap(colormap)
    color_image = colormap_fn(normalized_image)

    color_image = (color_image[:, :, :3] * 255).astype(np.uint8)
    color_image = torch.from_numpy(color_image).permute(2,0,1)

    return color_image.unsqueeze(0).float()

def input_padding(x, target_width=1280):

    _,_,_, W = x.shape

    pad_w = target_width - W
    pad_h = 0  
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))  

    return x_padded


def edge_aware_smoothness(rgb, depth):
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

def smoothness_loss(residual):
    """
    Compute a simple total variation loss over the residual.
    Assumes residual shape is (B, 1, H, W).
    """
    grad_x = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
    grad_y = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])
    return grad_x.mean() + grad_y.mean()


def train(args):

    rank, world_size = setup_distributed()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    if rank == 0:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            project='Depth-Completion',
            name=f'Fusion_{args.crop_size}_{args.batch_size}_{args.lr}_{args.epochs}',
            config=args,
            mode='online'
        )
    else:
        os.environ["WANDB_MODE"] = "offline"

    train_dataloader, len_train = get_dataloader(args, train=True, distributed=(world_size > 1), rank=rank, world_size=world_size)
    val_dataloader, _ = get_dataloader(args, train=False, distributed=(world_size > 1), rank=rank, world_size=world_size)

    model = __models__[args.model](args)

    #trainable_layers = ["cnet", "stem", "cost_agg", "context"] #"feature"

    for name, param in model.named_parameters():
        if "stereo_matching" in name:
            param.requires_grad = True
            #if any(layer in name for layer in trainable_layers):
            #    param.requires_grad = True
            #else:
            #    param.requires_grad = False
        else:
            param.requires_grad = True

    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=True) 

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer, len_train)

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

    #torch.autograd.set_detect_anomaly(True)

    for epoch_idx in range(start_epoch, args.epochs):
        
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        model.train()
        #model.module.stereo_matching.eval()
        progress_bar = tqdm(enumerate(train_dataloader), 
                            total=len(train_dataloader),
                            desc=f"Epoch {epoch_idx + 1}",
                            disable=(rank != 0))
        epoch_loss = 0.0

        for batch_idx, batch_sample in progress_bar:

            rgb_aug, rgb_left, rgb_right, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]
            if args.model == 'depth_fusion_swin':
                rgb = input_padding(rgb)
                stereo = input_padding(stereo)
                sparse = input_padding(sparse)

            optimizer.zero_grad()

            stereo_depth, init_pred, final_pred, r_depth, r_disp = model(rgb_aug, rgb_left, rgb_right, sparse, width)

            if args.model == 'depth_fusion_swin':
                rgb = rgb[:, :, :256, :1216]
                #confidence = confidence[:, :, :256, :1216]
                init_pred = init_pred[:, :, :256, :1216]
                final_pred = final_pred[:, :, :256, :1216]

            mask = (groundtruth > 1e-8)
            gt_disp = depth_to_disparity_train(groundtruth, width)
            mask_disp = gt_disp > 1e-8

            loss_disp = criterion_l2(stereo_depth[mask_disp], gt_disp[mask_disp]) + criterion_l1(stereo_depth[mask_disp], gt_disp[mask_disp])
            #loss_disp = criterion_l2(stereo_depth[mask], groundtruth[mask]) + criterion_l1(stereo_depth[mask], groundtruth[mask])
            loss_init = criterion_l2(init_pred[mask], groundtruth[mask]) + criterion_l1(init_pred[mask], groundtruth[mask]) 
            loss_final = criterion_l2(final_pred[mask], groundtruth[mask]) + criterion_l1(final_pred[mask], groundtruth[mask])
            loss = 0.6 * loss_final + 0.8 * loss_init + 0.8 * loss_disp #+ 0.3 * edge_aware_smoothness(rgb_left, final_pred)
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0, norm_type=2.0, foreach=None)
            optimizer.step()

            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss)
            reduced_loss /= world_size

            epoch_loss += reduced_loss.item()

            #if batch_idx % 200 == 0 and batch_idx != 0:
            #   metrics_eval = validate(rank, model, val_dataloader, metric_evaluator, args)         

            if rank == 0:
                progress_bar.set_postfix(loss=reduced_loss.item())
                
                wandb.log({
                    "batch_loss": reduced_loss.item(),
                    "disparity to depth loss": loss_disp.item(),
                    "init pred loss": loss_init.item(),
                    "final pred loss": loss_final.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm of this parameter's gradient
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                wandb.log({"grad_norm": total_norm})
                if batch_idx % 1500 == 0 and rank == 0:
                    """                    
                    print(f"Image RGB Min {rgb_aug.min()} Max {rgb_aug.max()}")
                    print(f"Image Left Min {rgb_left.min()} Max {rgb_left.max()}")
                    print(f"Image Right Min {rgb_right.min()} Max {rgb_right.max()}")
                    print(f"Sparse Min {sparse.min()} Max {sparse.max()}")
                    print(f"Groundtruth Min {groundtruth.min()} Max {groundtruth.max()}")
                    print(f"Groundtruth Disparity Min {gt_disp.min()} Max {gt_disp.max()}")
                    print(f"Stereo Depth Min {stereo_depth.min()} Max {stereo_depth.max()}")
                    print(f"Init Pred Min {init_pred.min()} Max {init_pred.max()}")
                    print(f"Final Pred Min {final_pred.min()} Max {final_pred.max()}")
                    print(f"Residual Depth Min {r_depth.min()} Max {r_depth.max()}")
                    print(f"Residual Disparity Min {r_disp.min()} Max {r_disp.max()}")
                    """
                    log_predictions(rank,
                                    rgb_left,
                                    rgb_right,
                                    groundtruth,
                                    gt_disp,
                                    stereo_depth,
                                    init_pred,
                                    final_pred,
                                    r_disp,
                                    r_depth,
                                    caption="Train Predictions")


        #if isinstance(lr_scheduler, optim.lr_scheduler.CosineAnnealingLR):
        #    lr_scheduler.step()

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
    total_metrics = {metric: 0.0 for metric in metric_evaluator.metrics}
    total_samples = 0  

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(rank != 0))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_sample in progress_bar:
            rgb_aug, rgb_left, rgb_right, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]

            if args.model == 'depth_fusion_swin':
                rgb = input_padding(rgb)
                stereo = input_padding(stereo)
                sparse = input_padding(sparse)

            stereo_depth, init_pred, final_pred, r_depth, r_disp = model(rgb_aug, rgb_left, rgb_right, sparse, width)
            gt_disp = depth_to_disparity_train(groundtruth, width)

            if args.model == 'depth_fusion_swin':
                stereo = stereo[:, :, :256, :1216]
                init_pred = init_pred[:, :, :256, :1216]
                final_pred = final_pred[:, :, :256, :1216]
                confidence = confidence[:, :, :256, :1216]

            # Evaluate metrics for the current batch
            batch_metrics = metric_evaluator.evaluate_metrics(final_pred, groundtruth) 
            batch_size = groundtruth.size(0)  

            for metric, value in batch_metrics.items():
                total_metrics[metric] += value.item() * batch_size 

            total_samples += batch_size

            avg_metrics = {metric: total / total_samples for metric, total in total_metrics.items()}
            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

            
            if batch_idx == 0 and rank == 0:
                """                
                print(f"Image RGB Min {rgb_aug.min()} Max {rgb_aug.max()}")
                print(f"Image Left Min {rgb_left.min()} Max {rgb_left.max()}")
                print(f"Image Right Min {rgb_right.min()} Max {rgb_right.max()}")
                print(f"Sparse Min {sparse.min()} Max {sparse.max()}")
                print(f"Groundtruth Min {groundtruth.min()} Max {groundtruth.max()}")
                print(f"Stereo Depth Min {stereo_depth.min()} Max {stereo_depth.max()}")
                print(f"Init Pred Min {init_pred.min()} Max {init_pred.max()}")
                print(f"Final Pred Min {final_pred.min()} Max {final_pred.max()}")
                print(f"Residual Depth Min {r_depth.min()} Max {r_depth.max()}")
                """

                log_predictions(rank,
                                rgb_left,   
                                rgb_right, 
                                groundtruth, 
                                gt_disp,
                                stereo_depth, 
                                init_pred, 
                                final_pred, 
                                r_disp,
                                r_depth, 
                                caption="Validation Predictions")
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
