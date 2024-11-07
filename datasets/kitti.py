import os
from .utils import get_kitti_paths

from torch.utils import data


class KittiDepthCompletion(data.Dataset):
    def __init__(self, data_path, crop_size, training=True):

        self.crop_size = crop_size
        self.stereo_filenames, self.sparse_filenames, self.gt_filenames = self.get_kitti_paths(data_path, training)
        self.training = training

        """
        assert os.path.exists(data_path)
        mode = 'train' if self.training == True else 'val'
        scene_date_list = sorted(os.listdir(os.path.join(data_path, mode)))

        for scene_id in scene_date_list:
            rgb_left_dir = os.path.join(rgb_dir)
        """

            
data_path = './Data'
crop_size = (352, 1216)
training = True
kitti_dataset = KittiDepthCompletion(data_path=data_path, crop_size=crop_size, training=training)

