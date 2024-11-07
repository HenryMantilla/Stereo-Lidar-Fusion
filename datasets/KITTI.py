import os

from torch.utils import data
from glob import glob

class KittiDepthCompletion(data.Dataset):
    def __init__(self, data_path, mode, crop_size):

        self.data_path = data_path
        self.crop_size = crop_size
        self.mode = mode

        assert os.path.exists(data_path)
        scene_date_list = sorted(os.listdir(os.path.join(data_path, mode)))

        for scene_id in scene_date_list:

            rgb_left_dir = os.path.join(rgb_dir)

            



data_path = './data'
data_split = 'train'
crop_size = (352, 1216)
kitti_dataset = KittiDepthCompletion(data_path=data_path, mode=data_split, crop_size=crop_size)

