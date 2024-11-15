#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
from utils.image_utils import load_img
import json

import os
from typing import Tuple, List
from dataclasses import dataclass

WARNED = False

@dataclass
class Intrinsics:
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float

    focal_xs: list
    focal_ys: list
    center_xs: list
    center_ys: list
    def convert_to_array(self):
        self.focal_xs = np.array(self.focal_xs, dtype=np.float64)
        self.focal_ys = np.array(self.focal_ys, dtype=np.float64)
        self.center_xs = np.array(self.center_xs, dtype=np.float64)
        self.center_ys = np.array(self.center_ys, dtype=np.float64)

    def scale(self, factor: float):
        self.convert_to_array()
        nw = round(self.width * factor)
        nh = round(self.height * factor)
        sw = nw / self.width
        sh = nh / self.height
        self.focal_x *= sw
        self.focal_y *= sh
        self.center_x *= sw
        self.center_y *= sh
        self.width = int(nw)
        self.height = int(nh)
        self.focal_xs = self.focal_xs * sw
        self.focal_ys = self.focal_ys * sh
        self.center_xs = self.center_xs * sw
        self.center_ys = self.center_ys * sh

    def append(self, focal_x, focal_y, center_x, center_y):
        self.focal_xs.append(focal_x)
        self.focal_ys.append(focal_y)
        self.center_xs.append(center_x)
        self.center_ys.append(center_y)

    def __repr__(self):
        return (f"Intrinsics(width={self.width}, height={self.height}, "
                f"focal_x={self.focal_x}, focal_y={self.focal_y}, "
                f"center_x={self.center_x}, center_y={self.center_y})")

def loadCam(args, id, cam_info, resolution_scale):
    if cam_info.image_path is None:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=None, gt_alpha_mask=None,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  depth=cam_info.depth,
                  mask = cam_info.mask, K=cam_info.K, image_width = cam_info.width, image_height = cam_info.height)
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if cam_info.image is not None: 
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    else:
        image, mask = load_img(cam_info.image_path, white_background = cam_info.white_background)
        resized_image_rgb = PILtoTorch(image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  depth=cam_info.depth,
                  mask = cam_info.mask, K=cam_info.K, image_width = cam_info.width, image_height = cam_info.height)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )
