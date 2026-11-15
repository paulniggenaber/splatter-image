import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from .shared_dataset import SharedDataset

# Data directory = the parent folder of srn_cars/ (and srn_chairs/).
# Read from the environment so that every team member can use their own path:
#   export SHAPENET_DATASET_ROOT=$HOME/data/SRN
SHAPENET_DATASET_ROOT = os.environ.get("SHAPENET_DATASET_ROOT")
assert SHAPENET_DATASET_ROOT is not None, "Set the SHAPENET_DATASET_ROOT environment variable to the SRN Shapenet data folder"

class SRNDataset(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"

        self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
                                                                                   cfg.data.category,
                                                                                   self.dataset_name))

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        print(len(self.intrins))
        if cfg.data.subset != -1:
            self.intrins = self.intrins[:cfg.data.subset]

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.cfg.data.input_images == 1:
            self.test_input_idxs = [64]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [64, 128]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.intrins)

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

        if example_id in self.all_rgbs.keys():
            return      # already in RAM

        # Optional disk cache: one file per object instead of ~100 (train) or ~500
        # (val/test) small files. On a network drive, reading the small files takes
        # about 3 s per object, i.e. hours for the whole training set at every job
        # start. Set SRN_CACHE_DIR to switch it on; fill it once with
        # `python -m datasets.cache_srn` (or it fills itself on first use).
        cache_path = self._cache_path(example_id)
        if cache_path is not None and os.path.isfile(cache_path):
            cached = torch.load(cache_path)
            self.all_rgbs[example_id] = cached["rgbs"].float() / 255.0     # exactly what PILtoTorch gives
            self.all_world_view_transforms[example_id] = cached["world_view_transforms"]
            self.all_view_to_world_transforms[example_id] = cached["view_to_world_transforms"]
            self.all_full_proj_transforms[example_id] = cached["full_proj_transforms"]
            self.all_camera_centers[example_id] = cached["camera_centers"]
            return

        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        assert len(rgb_paths) == len(pose_paths)

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []

            cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))])

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)
            
            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])

            if cache_path is not None:
                # images as bytes (4x smaller); k/255 -> k is exact, so nothing is lost
                to_save = {
                    "rgbs": (self.all_rgbs[example_id] * 255.0).round().to(torch.uint8),
                    "world_view_transforms": self.all_world_view_transforms[example_id],
                    "view_to_world_transforms": self.all_view_to_world_transforms[example_id],
                    "full_proj_transforms": self.all_full_proj_transforms[example_id],
                    "camera_centers": self.all_camera_centers[example_id],
                }
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                tmp_path = cache_path + ".tmp{}".format(os.getpid())
                torch.save(to_save, tmp_path)
                os.replace(tmp_path, cache_path)     # atomic: never leaves a half-written file

    def _cache_path(self, example_id):
        cache_root = os.environ.get("SRN_CACHE_DIR")
        if not cache_root:
            return None
        # resolution and camera settings are part of the folder name, so a different
        # setting never picks up a stale cache
        tag = "srn_{}_{}px_fov{:.3f}_z{}-{}".format(self.cfg.data.category,
                                                   self.cfg.data.training_resolution,
                                                   self.cfg.data.fov,
                                                   self.cfg.data.znear, self.cfg.data.zfar)
        return os.path.join(cache_root, tag, self.dataset_name, example_id + ".pt")

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))

        self.load_example_id(example_id, intrin_path)
        if self.dataset_name == "train":
            frame_idxs = torch.randperm(
                    len(self.all_rgbs[example_id])
                    )[:self.imgs_per_obj]

            frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)

        else:
            input_idxs = self.test_input_idxs
            
            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 

        images_and_camera_poses = {
            "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
            "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
            "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
            "camera_centers": self.all_camera_centers[example_id][frame_idxs]
        }

        # True mirror plane of the object, expressed in the frame of the input camera.
        # SRN cars and chairs are stored aligned: the mirror plane is x = 0 in dataset
        # coordinates (checked on the data: camera on the x axis sees the car side-on).
        # The plane is all points p with n . p = d. Must be computed BEFORE the poses
        # are made relative to the first (input) camera.
        wvt0 = images_and_camera_poses["world_view_transforms"][0]   # world -> input camera, row-major
        sym_n = wvt0[0, :3].clone()                                  # dataset x axis seen from the input camera
        sym_d = (sym_n * wvt0[3, :3]).sum()                          # wvt0[3, :3] = object centre in camera coords
        sym_plane = torch.cat([sym_n, sym_d[None]]).float()          # (4,)

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
        images_and_camera_poses["sym_plane"] = sym_plane

        if self.dataset_name == "test":
            # Testing visits every object exactly once, so keeping all 251 views of
            # all objects cached only fills the RAM (704 test cars = ~35 GB).
            # (train and val are visited many times, so they stay cached.)
            for cache in [self.all_rgbs, self.all_world_view_transforms,
                          self.all_view_to_world_transforms, self.all_full_proj_transforms,
                          self.all_camera_centers]:
                del cache[example_id]

        return images_and_camera_poses