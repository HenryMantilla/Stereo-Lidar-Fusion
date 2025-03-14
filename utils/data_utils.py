import os
from glob import glob
import random

def get_kitti_paths(data_path):

    assert os.path.exists(data_path)

    scene_paths = []
    mode_dir = os.path.join(data_path)
    scene_date_list = sorted(os.listdir(mode_dir))
    
    for scene_date in scene_date_list:
        scene_date_path = os.path.join(mode_dir, scene_date)
        scene_ids = sorted(os.listdir(scene_date_path))
        
        for scene_id in scene_ids:
            scene_id_path = os.path.join(mode_dir, scene_date, scene_id)
            scene_paths.append(scene_id_path)

    return scene_paths


def get_kitti_files(data_path):

    scene_paths = get_kitti_paths(data_path)
    rgb_left_files, rgb_right_files, sparse_files, gt_files = [], [], [], []

    for scene in scene_paths:
        rgb_left_dir = os.path.join(scene, 'image_02', 'data')
        rgb_right_dir = os.path.join(scene, 'image_03', 'data')
        sparse_dir = os.path.join(scene, 'proj_depth', 'velodyne_raw', 'image_02')
        gt_dir = os.path.join(scene, 'proj_depth', 'groundtruth', 'image_02')

        rgb_left_files.extend(sorted(glob(os.path.join(rgb_left_dir, "*.png"))))
        rgb_right_files.extend(sorted(glob(os.path.join(rgb_right_dir, "*.png"))))
        sparse_files.extend(sorted(glob(os.path.join(sparse_dir, "*.png"))))
        gt_files.extend(sorted(glob(os.path.join(gt_dir, "*.png"))))

    assert len(rgb_left_files) == len(rgb_right_files) == len(sparse_files) == len(gt_files), \
    "The number of images must be the same."
    
    """
    num_files = len(sparse_files)
    num_to_keep = int(num_files * 0.2)
    selected_indices = random.sample(range(num_files), num_to_keep)
    selected_indices.sort()

    # Keep only selected files
    sparse_files = [sparse_files[i] for i in selected_indices]
    gt_files = [gt_files[i] for i in selected_indices]
    rgb_left_files = [rgb_left_files[i] for i in selected_indices]
    rgb_right_files = [rgb_right_files[i] for i in selected_indices]"
    """
    
    return rgb_left_files, rgb_right_files, sparse_files, gt_files 