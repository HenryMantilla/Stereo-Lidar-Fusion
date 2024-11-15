import os
import torch

from utils import data_utils
from utils import frame_utils

from loguru import logger
from torch.utils import data
from torchvision.transforms import v2 as transforms


class KittiDepthCompletion(data.Dataset):
    def __init__(self, data_path, crop_size, training=True):

        mode = 'train' if training else 'val'
        split_path = os.path.join(data_path, mode)

        self.crop_size = crop_size
        self.stereo_filenames, self.sparse_filenames, self.gt_filenames = data_utils.get_kitti_files(split_path)
        self.training = training

        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False)
        ])

    def __len__(self):
        return len(self.stereo_filenames)
    
    def __getitem__(self, idx):
        
        stereo_disp = frame_utils.read_disp(self.stereo_filenames[idx])
        sparse_disp = frame_utils.depth_to_disparity(frame_utils.read_depth(self.sparse_filenames[idx]))
        gt_disp = frame_utils.depth_to_disparity(frame_utils.read_depth(self.gt_filenames[idx]))

        stereo_normalized, sparse_normalized, gt_normalized = list(map(lambda x: frame_utils.normalize_image(x), [stereo_disp, sparse_disp, gt_disp]))

        if self.training:
            stereo_crop, sparse_crop, gt_crop = list(map(lambda x: frame_utils.crop_center(x, self.crop_size),
                                                                    [stereo_normalized, sparse_normalized, gt_normalized]))
        
        stereo, sparse, gt = list(map(lambda x: self.transform(x), [stereo_crop, sparse_crop, gt_crop] ))
        
        return stereo, sparse, gt
    

def fetch_dataloader(args):

    if args.dataset == 'kitti':
        
        dataset = KittiDepthCompletion(args.data_path, args.crop_size, args.training)
        logger.info(f"Loading {len(dataset)} images from KITTI Depth Completion dataset.")

    
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    

    return dataloader




"""
data_path = './Data'
crop_size = (256, 512)
training = True
kitti_dataset = KittiDepthCompletion(data_path=data_path, crop_size=crop_size, training=training)

stereo, sparse, gt = kitti_dataset[0]


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(stereo.permute(1,2,0), cmap='magma')
axes[0].set_title('Stereo Depth')

axes[1].imshow(sparse.permute(1,2,0), cmap='magma')
axes[1].set_title('LiDAR')

axes[2].imshow(gt.permute(1,2,0), cmap='magma')
axes[2].set_title('GT depth map')

plt.show()
"""