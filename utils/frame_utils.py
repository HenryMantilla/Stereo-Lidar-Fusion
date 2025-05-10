import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from scipy.ndimage import label

def depth_to_disparity(depth): 

    baseline = 0.54
    width_to_focal = dict()

    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.8560
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046
    width_to_focal[1238] = 718.3351

    depth = depth.astype(np.float32) 
    invalid_depth = depth <= 0
    _, width = depth.shape[:2]
    focal_length = width_to_focal[width]

    disparity = (focal_length * baseline) / (depth + 1e-8)

    disparity[invalid_depth] = 0

    return disparity

def depth_to_disparity_train(depth, width):
    baseline = 0.54
    width_to_focal = {
        1242: 721.5377,
        1241: 718.8560,
        1224: 707.0493,
        1226: 708.2046,
        1238: 718.3351
    }
    # Ensure depth is float32 tensor
    depth = depth.float()
    focal_length = width_to_focal.get(width, width_to_focal[1242])
    # Create a mask for invalid depth values (<=0)
    invalid_depth = depth <= 0
    disparity = (focal_length * baseline) / (depth + 1e-8)
    disparity[invalid_depth] = 0
    
    return disparity


def disparity_to_depth_kitti(disparity):
    baseline = 0.54
    width_to_focal = dict()

    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.8560
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046
    width_to_focal[1238] = 718.3351

    disparity = disparity.astype(np.float32) 
    _, width = disparity.shape[:2]
    focal_length = width_to_focal[width]
    
    depth = (focal_length * baseline) / (disparity + 1e-8)
    depth[disparity <= 0] = 0

    return depth

def disparity_to_depth(disparity, widths):
    baseline = 0.54
    width_to_focal = {
        1242: 721.5377,
        1241: 718.8560,
        1224: 707.0493,
        1226: 708.2046,
        1238: 718.3351
    }

    assert widths.shape[0] == disparity.shape[0], "Widths tensor must match batch dimension of disparity"

    focal_lengths = torch.tensor([
        width_to_focal.get(width.item(), width_to_focal[1242]) 
        for width in widths
    ], dtype=disparity.dtype, device=disparity.device).view(-1, 1, 1, 1)
    
    depth = (focal_lengths * baseline) / (disparity + 1e-8)
    depth[disparity <= 0] = 0

    return depth

def normalize_image(image):

    min_val = image.min()
    max_val = image.max()
    normalized_img = (image - min_val) / (max_val - min_val)

    return normalized_img

def read_rgb(image_path):

    image = np.array(Image.open(image_path)).astype(np.float32)

    return image

def read_disp(image_path):

    disp = np.array(Image.open(image_path), dtype=np.uint16) / 256.0

    return disp

def read_depth(image_path):

    depth = np.array(Image.open(image_path), dtype=np.uint16) / 256.0
    
    return np.expand_dims(depth, axis=-1)

def crop_fixed_size(image, crop_size):

    H, W, _ = image.shape

    crop_H, crop_W = crop_size
    start_x, start_y = (W - crop_W) // 2, (H - crop_H) // 2

    cropped_img = image[start_y:start_y + crop_H, start_x:start_x + crop_W]

    return cropped_img

def crop_bottom_center(image, crop_size):

    if isinstance(image, np.ndarray):
        tensor_img = TF.to_tensor(image)
    else:
        tensor_img = image

    _, h, w = tensor_img.shape
    crop_h, crop_w = crop_size

    assert crop_h <= h and crop_w <= w, "Crop size exceeds the image dimensions"

    left = (w - crop_w) // 2  
    top = h - crop_h  
    
    cropped_img = TF.crop(tensor_img, top, left, crop_h, crop_w)

    return cropped_img.permute(1, 2, 0).numpy()