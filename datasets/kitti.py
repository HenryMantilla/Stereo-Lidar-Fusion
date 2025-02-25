import os
import torch
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
        self.rgb_left_filenames, self.rgb_right_filenames, self.sparse_filenames, self.gt_filenames = data_utils.get_kitti_files(split_path)
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
        return len(self.sparse_filenames)
    
    def __getitem__(self, idx):

        rgb_left = frame_utils.read_rgb(self.rgb_left_filenames[idx])     
        rgb_right = frame_utils.read_rgb(self.rgb_right_filenames[idx])   
        sparse_disp = frame_utils.read_depth(self.sparse_filenames[idx]) #frame_utils.depth_to_disparity(frame_utils.read_depth(self.sparse_filenames[idx]))
        gt_disp = frame_utils.read_depth(self.gt_filenames[idx]) #frame_utils.depth_to_disparity(frame_utils.read_depth(self.gt_filenames[idx]))

        width = gt_disp.shape[1]

        rgb_left_crop = frame_utils.crop_bottom_center(rgb_left, self.crop_size)
        rgb_right_crop = frame_utils.crop_bottom_center(rgb_right, self.crop_size)
        sparse_crop = frame_utils.crop_bottom_center(sparse_disp, self.crop_size)
        gt_crop = frame_utils.crop_bottom_center(gt_disp, self.crop_size)

        if self.training:
            rgb_left_clean = self.transform_train(rgb_left_crop)   
            rgb_right_clean = self.transform_train(rgb_right_crop) 
            sparse = self.transform_train(sparse_crop)
            gt = self.transform_train(gt_crop)

            rgb_left_aug = rgb_left_clean.clone()

            brightness = np.random.uniform(0.6, 1.2)
            contrast = np.random.uniform(0.6, 1.2)
            saturation = np.random.uniform(0.6, 1.2)
            hue = np.random.uniform(-0.1, 0.1)

            rgb_left_aug = transforms.functional.adjust_brightness(rgb_left_aug, brightness)
            rgb_left_aug = transforms.functional.adjust_contrast(rgb_left_aug, contrast)
            rgb_left_aug = transforms.functional.adjust_saturation(rgb_left_aug, saturation)
            rgb_left_aug = transforms.functional.adjust_hue(rgb_left_aug, hue)

            rgb_left_aug = transforms.functional.normalize(rgb_left_aug, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            #degree = float(np.random.uniform(-5.0, 5.0))
            #rgb_left_aug = transforms.functional.rotate(
            #    rgb_left_aug, angle=degree, interpolation=transforms.InterpolationMode.BILINEAR
            #)

            #if random.random() > 0.5:
            #    rgb_left_aug = transforms.functional.hflip(rgb_left_aug)

            return rgb_left_aug, rgb_left_clean, rgb_right_clean, sparse, gt, width

        else:
            rgb_left_clean = self.transform_val(rgb_left_crop)
            rgb_right_clean = self.transform_val(rgb_right_crop)
            sparse = self.transform_val(sparse_crop)
            gt = self.transform_val(gt_crop)

            rgb_left_aug = rgb_left_clean
            rgb_left_aug = transforms.functional.normalize(rgb_left_aug, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            return rgb_left_aug, rgb_left_clean, rgb_right_clean, sparse, gt, width

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