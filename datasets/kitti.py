import os
import torch
import random
import numpy as np

from utils import data_utils
from utils import frame_utils

from loguru import logger
from torch.utils import data
from torchvision.transforms import v2 as transforms

from torch.utils.data.distributed import DistributedSampler


class KittiDepthCompletion(data.Dataset):
    def __init__(self, data_path, crop_size, training=True):

        mode = 'train' if training else 'val'
        split_path = os.path.join(data_path, mode)

        self.crop_size = crop_size
        self.rgb_filenames, self.stereo_filenames, self.sparse_filenames, self.gt_filenames = data_utils.get_kitti_files(split_path)
        self.training = training

        self.transform_train = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False)
        ])

        self.transform_val = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False)
        ])

    def __len__(self):
        return len(self.stereo_filenames)
    
    def __getitem__(self, idx):
        
        rgb = frame_utils.read_rgb(self.rgb_filenames[idx])
        stereo_disp = frame_utils.disparity_to_depth_kitti(frame_utils.read_disp(self.stereo_filenames[idx]))
        sparse_disp = frame_utils.read_depth(self.sparse_filenames[idx])
        gt_disp = frame_utils.read_depth(self.gt_filenames[idx])

        width = gt_disp.shape[1]

        if self.training: 
            rgb_crop, stereo_crop, sparse_crop, gt_crop = [frame_utils.crop_bottom_center(x, self.crop_size) for x in [rgb, stereo_disp, sparse_disp, gt_disp]]
            
            #sparse_max_depth = np.max(sparse_crop)
            #sparse_crop_ip = frame_utils.fill_in_fast(np.squeeze(sparse_crop, axis=-1), max_depth=sparse_max_depth, blur_type='gaussian')
            
            if random.random() > 0.5:
                rgb_crop = np.flip(rgb_crop, axis=1).copy()
                stereo_crop = np.flip(stereo_crop, axis=1).copy()
                sparse_crop = np.flip(sparse_crop, axis=1).copy()
                gt_crop = np.flip(gt_crop, axis=1).copy()
        
            if random.random() > 0.5:
                rgb_crop = np.flip(rgb_crop, axis=0).copy()
                stereo_crop = np.flip(stereo_crop, axis=0).copy()
                sparse_crop = np.flip(sparse_crop, axis=0).copy()
                gt_crop = np.flip(gt_crop, axis=0).copy()

        else:
            rgb_crop, stereo_crop, sparse_crop, gt_crop = list(map(lambda x: frame_utils.crop_bottom_center(x, self.crop_size),
                                                                   [rgb, stereo_disp, sparse_disp, gt_disp]))
            #sparse_max_depth = np.max(sparse_crop)
            #sparse_crop_ip = frame_utils.fill_in_fast(np.squeeze(sparse_crop, axis=-1), max_depth=sparse_max_depth, blur_type='gaussian')
            rgb, stereo, sparse, gt = [self.transform_val(x) for x in [rgb_crop, stereo_crop, sparse_crop, gt_crop]] #rgb_crop, stereo_crop, sparse_crop, gt_crop
            stereo_lidar = torch.where(sparse > 1e-8, sparse, stereo)
        
        if self.training:

            rgb, stereo, sparse, gt = [self.transform_train(x) for x in [rgb_crop, stereo_crop, sparse_crop, gt_crop]]

            brightness = np.random.uniform(0.6, 1.2)
            contrast = np.random.uniform(0.6, 1.2)
            saturation = np.random.uniform(0.6, 1.2)
            hue = np.random.uniform(-0.1, 0.1)

            rgb = transforms.functional.adjust_brightness(rgb, brightness)
            rgb = transforms.functional.adjust_contrast(rgb, contrast)
            rgb = transforms.functional.adjust_saturation(rgb, saturation)
            rgb = transforms.functional.adjust_hue(rgb, hue)

            stereo_lidar = torch.where(sparse > 1e-8, sparse, stereo)

            degree = float(np.random.uniform(-5.0, 5.0))
            rgb = transforms.functional.rotate(rgb, angle=degree, interpolation=transforms.InterpolationMode.BILINEAR)
            #stereo = transforms.functional.rotate(stereo, angle=degree, interpolation=transforms.InterpolationMode.NEAREST)
            stereo_lidar = transforms.functional.rotate(stereo_lidar, angle=degree, interpolation=transforms.InterpolationMode.NEAREST)
            sparse = transforms.functional.rotate(sparse, angle=degree, interpolation=transforms.InterpolationMode.NEAREST)
            gt = transforms.functional.rotate(gt, angle=degree, interpolation=transforms.InterpolationMode.NEAREST)
        
        return rgb, stereo_lidar, sparse, gt, width
    

def get_dataloader(args, train=True, distributed=False, rank=0, world_size=1):

    if not train:
        args.batch_size = 1
        args.crop_size = (256, 1216)

    dataset = KittiDepthCompletion(args.data_path, args.crop_size, train)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train) if distributed else None

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and train),  
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=train
    )

    if rank == 0:
        logger.info(f"Loading {len(dataset)} images from KITTI Depth Completion dataset.")
        if distributed:
            logger.info(f"Using DistributedSampler with {world_size} processes.")

    return dataloader, len(dataset)