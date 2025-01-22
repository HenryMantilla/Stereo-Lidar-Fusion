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
    #depth[disparity <= 0] = 0

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

    return normalize_image(image)

def read_disp(image_path):

    disp = np.array(Image.open(image_path), dtype=np.uint16) / 256.0

    return disp

def read_depth(image_path):

    depth = np.array(Image.open(image_path), dtype=np.uint16) / 256.0
    
    return np.expand_dims(depth, axis=-1)

def resize_image(image, target_size=(1024, 1024)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

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


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_13 = np.ones((13, 13), np.uint8)
FULL_KERNEL_25 = np.ones((25, 25), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64
    
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # fill zero value
    mask = (depth_map <= 0.1)
    if np.sum(mask) != 0:
        labeled_array, num_features = label(mask)
        for i in range(num_features):
            index = i + 1
            m = (labeled_array == index)
            m_dilate1 = cv2.dilate(1.0*m, FULL_KERNEL_7)
            m_dilate2 = cv2.dilate(1.0*m, FULL_KERNEL_13)
            m_diff = m_dilate2 - m_dilate1
            v = np.mean(depth_map[m_diff>0])
            depth_map = np.ma.array(depth_map, mask=m_dilate1, fill_value=v)
            depth_map = depth_map.filled()
            depth_map = np.array(depth_map)
    #else:
    #    depth_map = depth_map

    depth_map = np.expand_dims(depth_map, 2)

    return depth_map