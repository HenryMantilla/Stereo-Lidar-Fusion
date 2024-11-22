import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image

def depth_to_disparity(depth): #Generalize for other datasets in case

    baseline = 0.54
    width_to_focal = dict()

    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.8560
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046
    width_to_focal[1238] = 718.3351

    depth[depth == 0] = -1
    _, width = depth.shape[:2]
    focal_length = width_to_focal[width]

    disparity = (focal_length * baseline) / depth

    return disparity


def disparity_to_depth(disparity):

    baseline = 0.54
    width_to_focal = dict()

    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.8560
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046
    width_to_focal[1238] = 718.3351

    _, width = disparity.shape[:2]
    focal_length = width_to_focal[width]

    disparity = (focal_length * baseline) / (disparity + 1e8)

    return disparity


def normalize_image(image):

    min_val = image.min()
    max_val = image.max()
    normalized_img = (image - min_val) / (max_val - min_val)

    return normalized_img


def read_rgb(image_path):

    image = np.array(Image.open(image_path)).astype(np.float32)

    return normalize_image(image)

def read_disp(image_path):

    disp = np.array(Image.open(image_path)).astype(np.float32)
    valid_disp = disp[:,:,:3].mean(axis=-1)

    return valid_disp

def read_depth(image_path):

    depth = Image.open(image_path)
    depth = np.array(depth).astype(np.float32) / 256.0 #(H, W)
    
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

    return cropped_img


def invert_depth(depth):
    max_val = depth.max()
    inv_depth = depth - max_val

    return inv_depth

def get_valid_mask(input):
    return (input > 0).detach()