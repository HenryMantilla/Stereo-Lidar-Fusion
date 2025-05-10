import os
import cv2
import wandb
import argparse
import numpy as np
import torchvision.transforms as transforms
import matplotlib.cm as cm

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import Image
from utils import MetricEvaluator
from datasets import __datasets__, get_dataloader
from models import __models__

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    return rank, world_size

def apply_colormap(image_tensor, colormap='magma'):

    if image_tensor.is_sparse:
        image_tensor = image_tensor.to_dense()

    image_tensor = image_tensor.squeeze()
    if image_tensor.dim() != 2:
        image_tensor = image_tensor.reshape(image_tensor.shape[-2:])
    image_np = image_tensor.float().cpu().numpy()

    normalized_image = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-6)
    colormap_fn = matplotlib.colormaps[colormap]
    color_image = colormap_fn(normalized_image)
    color_image = (color_image[..., :3] * 255).astype(np.uint8)

    if color_image.ndim == 4:
        color_image = np.squeeze(color_image, axis=0)
    color_tensor = torch.from_numpy(color_image).permute(2, 0, 1)
    return color_tensor.unsqueeze(0).float()

"""
def tensor_to_wandb_fig(t, cmap="Spectral"):

    arr = t.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    im  = ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    img = wandb.Image(fig)          
    plt.close(fig)                  
    return img
"""

def tensor_to_wandb_fig(
        t,
        cmap      = "Spectral",
        bar_px    = 20,      # ancho de la barra en píxeles
        pad_px    = 8,       # espacio entre imagen y barra en píxeles
        dpi       = 200,
        lw_border = 0.5      # grosor del contorno de la barra
) -> wandb.Image:
    """
    Muestra un tensor (H×W) con una colorbar que:
      • tiene la MISMA altura que la imagen,
      • ancho fijo `bar_px`,
      • separación fija `pad_px`,
      • contorno fino (`lw_border`),
      • y números en la colorbar escalados con el tamaño de la figura.
    """
    # --- Convert tensor to numpy array ---
    if isinstance(t, torch.Tensor):
        arr = t.squeeze().detach().cpu().numpy()
    else:
        arr = np.asarray(t).squeeze()
    H, W = arr.shape

    # --- Figure size in inches so that figure has exact pixel dims at given dpi ---
    fig_w = (W + bar_px + pad_px) / dpi
    fig_h = H / dpi

    # --- Create figure with your dpi ---
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(arr, cmap=cmap, vmin=arr.min(), vmax=arr.max())
    ax.axis("off")

    # --- Append colorbar with exact height matching the image ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size=bar_px / dpi,   # width in inches
        pad = pad_px / dpi   # pad in inches
    )
    cbar = plt.colorbar(im, cax=cax)

    # --- Scale tick labels proportional to figure height ---
    # Here we choose a base of 6 points at H=200px, scaling linearly
    base_px = 200
    base_font_pt = 6
    font_pt = max(4, int(base_font_pt * (H / base_px)))
    cbar.ax.tick_params(labelsize=font_pt)

    # --- Thin border around the colorbar ---
    cbar.outline.set_linewidth(lw_border)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(lw_border)

    fig.tight_layout(pad=0)
    wandb_img = wandb.Image(fig)
    plt.close(fig)
    return wandb_img

def sparse_abs_err(abs_err, gt_depth, use_nan=True):
    #abs_err  = torch.as_tensor(abs_err)
    #gt_depth = torch.as_tensor(gt_depth)

    mask = (~torch.isnan(gt_depth)) if torch.isnan(gt_depth).any() else (gt_depth > 0)
    #fill = torch.nan if use_nan else 0.0
    fill = 0.0

    return abs_err.where(mask, torch.full_like(abs_err, fill))

def validate(rank, model, val_loader, metric_evaluator, args):

    total_metrics = {metric: 0.0 for metric in metric_evaluator.metrics}
    total_samples = 0  

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(rank != 0))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_sample in progress_bar:
            #rgb_aug, rgb_left, rgb_right, sparse, groundtruth, width = [x.cuda(non_blocking=True) for x in batch_sample]
            processed_batch = []
            for x in batch_sample:
                if isinstance(x, torch.Tensor):
                    processed_batch.append(x.cuda(non_blocking=True))
                else:
                    processed_batch.append(x)
            rgb_aug, rgb_left, rgb_right, sparse, groundtruth, width = processed_batch
            stereo_depth, init_pred, final_pred, r_depth, r_disp = model(rgb_aug, rgb_left, rgb_right, sparse, width)

            if batch_idx % 50 == 0 and rank==0:
                wandb.init(
                project='Depth-Completion-Test',
                name='Test',
                mode='online'
                )

                #print("Filename:", filename_left)
                # Process the guide tensor.
                # Expected guide shape: [1, 48, H, W] or similar.
                """
                guide_cpu = guide.detach().cpu().squeeze(0)  # shape becomes [48, H, W] (or possibly [48, 1, H, W])
                guide_images = []
                for ch in range(guide_cpu.size(0)):
                    # For each channel, pass the tensor directly to apply_colormap.
                    # The function will squeeze out any extra dimensions.
                    channel_tensor = guide_cpu[ch]
                    color_channel = apply_colormap(channel_tensor, colormap='Spectral')
                    # Convert the result from [1, 3, H, W] to HWC format for wandb.
                    color_channel_np = color_channel.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                    guide_images.append(wandb.Image(color_channel_np, caption=f"Guide Channel {ch}"))
                """
                # Process and colorize the other outputs.
                outputs = {
                    "Stereo Depth": stereo_depth,
                    "Init Pred": init_pred,
                    "R_Depth": r_depth,
                    "R_Disp": r_disp,
                    "Final Pred": final_pred,
                    "GT": groundtruth,
                }
                logged_outputs = {}
                for key, tensor in outputs.items():
                    #color_img = tensor_to_wandb_fig(tensor) colorbar}
                    color_img = apply_colormap(tensor, colormap='Spectral_r')
                    color_img_np = color_img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                    logged_outputs[key] = wandb.Image(color_img, caption=key)
                    #logged_outputs[key] = color_img colorbar
                
                wandb.log({
                    **logged_outputs
                })
                wandb.log({"Right": wandb.Image(rgb_right),
                "Left": wandb.Image(rgb_left)})

                #os.makedirs("depth_preds", exist_ok=True)
    
                # Detach final_pred, move to CPU, and remove the batch dimension if present.
                final_pred_np = final_pred.detach().cpu().squeeze().numpy()
        

            batch_metrics = metric_evaluator.evaluate_metrics(final_pred, groundtruth) 
            batch_size = groundtruth.size(0)  

            for metric, value in batch_metrics.items():
                total_metrics[metric] += value.item() * batch_size 

            total_samples += batch_size

            avg_metrics = {metric: total / total_samples for metric, total in total_metrics.items()}
            progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in avg_metrics.items()})

    # Distributed reduction if applicable
    if dist.is_initialized():
        for metric in total_metrics:
            metric_tensor = torch.tensor(total_metrics[metric], dtype=torch.float32, device="cuda")
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            total_metrics[metric] = metric_tensor.item()

        total_samples_tensor = torch.tensor(total_samples, dtype=torch.float32, device="cuda")
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = int(total_samples_tensor.item())

    avg_metrics = {metric: total / total_samples for metric, total in total_metrics.items()}

    return avg_metrics

def save_images(output_path, depth_tensor, colormap='plasma'):

    color_image = apply_colormap(depth_tensor, colormap)
    image_pil = transforms.ToPILImage()(color_image.squeeze(0))
    image_pil.save(output_path)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

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
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
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

# Example usage
if __name__ == "__main__":

    args = parse_args()
    rank, world_size = setup_distributed()
    val_dataloader, _ = get_dataloader(args, train=False, distributed=(world_size > 1), rank=rank, world_size=world_size)
    model =  __models__[args.model](args)
    model = model.cuda()
    checkpoint = torch.load("/ibex/user/perezpnf/henry/Stereo-Lidar-Fusion/checkpoints/crop_256x1216_lr_1e-3_adamW_test4/checkpoint_epoch_08.ckpt", map_location='cuda')
    model.load_state_dict(checkpoint['model'])

    metrics = ["mae_metric", "imae_metric", "rmse_metric", "irmse_metric"]
    metric_evaluator = MetricEvaluator(metrics)
    metrics_eval = validate(rank, model, val_dataloader, metric_evaluator, args)
    cleanup()
