import os
from glob import glob


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
    stereo_files, sparse_files, gt_files = [], [], []

    for scene in scene_paths:
        stereo_dir = os.path.join(scene, 'image_02', 'data') #Temporal, remeber change to stereo disparity
        sparse_dir = os.path.join(scene, 'proj_depth', 'velodyne_raw', 'image_02')
        gt_dir = os.path.join(scene, 'proj_depth', 'groundtruth', 'image_02')

        stereo_files.extend(sorted(glob(os.path.join(stereo_dir, "*.png"))))
        sparse_files.extend(sorted(glob(os.path.join(sparse_dir, "*.png"))))
        gt_files.extend(sorted(glob(os.path.join(gt_dir, "*.png"))))

    assert len(stereo_files) == len(sparse_files) == len(gt_files), \
    "The number of images must be the same."

    return stereo_files, sparse_files, gt_files 