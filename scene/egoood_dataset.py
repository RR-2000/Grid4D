import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm
import json 

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

class EgoOodDataset(Dataset):
    def __init__(
        self,
        datadir, 
        split,
        downsample=1,
        start_frame=2520,
        end_frame=3840, #exclude
        temporal_stride=2, #60->30fps
        imgdir='images_fps60_TV_undistort',
        posedir='vlg_colmap/000600/undistort_dense/sparse',
        splitdir='splits/perframe/3_static',
        load_image_on_the_fly = False,
    ):
        self.datadir, self.split, self.imgdir = datadir, split, imgdir  #bike-release
        self.start_frame, self.end_frame = start_frame, end_frame
        self.temporal_stride = temporal_stride
        self.downsample = downsample
        self.load_image_on_the_fly = load_image_on_the_fly

        self.transform = torchvision.transforms.ToTensor()

        extrinsics = read_extrinsics_binary(os.path.join(datadir, posedir, 'images.bin'))
        imgname2RT = {}
        for ex in extrinsics.values():
            R = np.transpose(qvec2rotmat(ex.qvec)) #transpose here
            T = np.array(ex.tvec)
            imgname2RT[ex.name] = (R, T)

        intrinsics = read_intrinsics_binary(os.path.join(datadir, posedir, 'cameras.bin'))
        assert len(intrinsics) == 1, 'Only one camera intrinsic is supported'
        intrinsic = list(intrinsics.values())[0]
        uid = intrinsic.id
        height, width = intrinsic.height, intrinsic.width
        assert intrinsic.model == 'PINHOLE'
        focal_length_x = intrinsic.params[0]
        focal_length_y = intrinsic.params[1]
        self.FovY = focal2fov(focal_length_y, height)
        self.FovX = focal2fov(focal_length_x, width)

        self.image_paths, self.image_poses, self.image_times, self.image_names = [], [], [], []
        for frame in range(start_frame, end_frame, self.temporal_stride):
            split = json.load(open(os.path.join(datadir, splitdir, f'{frame:06d}.json')))
            if self.split=='train':
                imgnames = split['3dgs-train']+split['ego-train']
            elif self.split=='test':
                imgnames = split['test']
            else:
                raise NotImplementedError
            for imgname in imgnames:
                self.image_paths.append(os.path.join(self.datadir, self.imgdir, f'{frame:06d}/images/{imgname}'))
                R, T = imgname2RT[imgname]
                self.image_poses.append((R, T))
                t = (frame-start_frame)/(end_frame-start_frame)
                self.image_times.append(t)
                self.image_names.append(f'frame{frame:06d}_view{imgname[4:-4]}')
                # print(image_name)

        print('done dataset')
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        imgname = self.image_names[index]
        if self.downsample > 1:
            img = img.resize((img.width//self.downsample, img.height//self.downsample))
        # img = self.transform(img)

        R,T = self.load_pose(index)
        
        # return img, self.image_poses[index], self.image_times[index], imgname
        return CameraInfo(uid=index, R=R, T=T, FovY=self.FovY, FovX=self.FovX, image=img,
                            image_path=self.image_paths[index] if not self.load_image_on_the_fly else None, image_name=imgname, width=img.size[0], height=img.size[1],
                            fid = self.image_times[index], mask=None, white_background = True, K=None) #TODO

    def load_pose(self, index):
        return self.image_poses[index]

