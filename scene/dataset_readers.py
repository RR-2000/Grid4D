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

import os
import sys
from PIL import Image
import torch
from typing import NamedTuple, Optional
import torchvision.transforms as transforms
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal,  getProjectionMatrix, ndc2Pix
import numpy as np
import json
import imageio
import tempfile
import trimesh
import uuid
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON, Intrinsics
from utils.image_utils import load_img
from tqdm import tqdm
from utils.camera_utils_multinerf import generate_interpolated_path


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mask: Optional[np.array] = None
    K: Optional[np.array] = None
    white_background: Optional[bool] = False


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    if 'vrig' in path:
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.5
    elif 'interp' in path:
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    elif 'nerf' in path:
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
        print("Assuming NeRF-DS dataset!")
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)
    print(len(selected_time))

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # ply_path = os.path.join(path, "points3D_downsample.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Find sfm point cloud:", ply_path)

    try:
        pcd = fetchPly(ply_path)
        print("Load sfm point cloud from:", ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted([a for a in glob(os.path.join(path, '*')) if os.path.isdir(a)])
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = os.path.join(video_paths[i], "images")
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def format_infos(dataset):
    # loading
    cameras = []
    for idx, (image, poses, time) in enumerate(tqdm(dataset, desc="Loading Neu3D")):
        image_path = dataset.image_paths[idx]
        image_name = '%04d.png' % idx
        # matrix = np.linalg.inv(np.array(pose))
        R, T = poses
        FovX = focal2fov(dataset.focal[0], image.size[0])
        FovY = focal2fov(dataset.focal[0], image.size[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            fid = time))

    return cameras

def readPlenopticVideoDataset(datadir, eval, num_images, hold_id=[0]):

    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    print("Find:", ply_path, "PCD:", pcd.points.shape)

    from scene.neu3d import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset)
    test_cam_infos = format_infos(test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readBrics(datadir, split, start_t: int = 0, num_t: int = 1, downsample: int = 1, white_background: bool = True, opencv_camera=True, load_image_on_the_fly = False):
    # per_cam_poses, intrinsics, cam_ids = load_brics_poses(datadir, downsample=downsample, split=split, opencv_camera=True)
    assert split in ['train', 'test', 'org']

    # load meta data
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']
    w, h = int(frames[0]['w']), int(frames[0]['h'])

    # load intrinsics
    intrinsics = Intrinsics(w, h, frames[0]['fl_x'], frames[0]['fl_y'], frames[0]['cx'], frames[0]['cy'], [], [], [], [] )
    for i in range(0, len(frames)):
        intrinsics.append(frames[i]['fl_x'], frames[i]['fl_y'], frames[i]['cx'], frames[i]['cy'])
    intrinsics.scale(1/downsample)

    # load poses
    cam_ids, poses = [], []
    for i in list(range(0, len(frames))):
        pose = np.array(frames[i]['transform_matrix'])
        if opencv_camera: # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            pose[:3, 1:3] *= -1
        poses.append(pose)
        cam_ids.append(frames[i]['file_path'].split('/')[-2])
    per_cam_poses = np.stack(poses)

    # load images and parse cameras
    cam_infos = []
    camera_dict = {}
    uid = 0
    for cam_idx in range(len(cam_ids)):
        cam_name = cam_ids[cam_idx]
        for j in tqdm(range(start_t, start_t+num_t), desc=f'Loading {split} data ({cam_idx}/{len(cam_ids)})'):
            img_path = os.path.join(datadir, "frames_1", cam_name,  f"{j:08d}.png")
            # per_cam_imgs.append(img_path)
            timestamp = j-start_t
            image_name = os.path.join(cam_name, f"{j:08d}") #Path(os.path.join(f"{cam_name}_{j:06d}").stem

            # load image and mask
            image, mask = load_img(img_path, downsample = downsample, white_background = white_background)
            
            # prep camera parameters
            # cam_idx = idx
            FovY = focal2fov(intrinsics.focal_ys[cam_idx], intrinsics.height)
            FovX = focal2fov(intrinsics.focal_xs[cam_idx], intrinsics.width)
            w2c = np.linalg.inv(np.array(per_cam_poses[cam_idx]))
            R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]

            K = np.array([[
                intrinsics.focal_xs[cam_idx], 0, intrinsics.center_xs[cam_idx]],
                [0, intrinsics.focal_ys[cam_idx], intrinsics.center_ys[cam_idx]],
                [0, 0, 1]]
            )
            cam_info = CameraInfo(uid=uid, fid=timestamp/float(num_t), R=R, T=T, FovY=FovY, FovX=FovX, K=K,
                image = image if not load_image_on_the_fly else None, mask = mask if not load_image_on_the_fly else None, 
                image_path=img_path, image_name=image_name, 
                width=image.size[0], height=image.size[1], white_background = white_background)
            uid += 1
            if timestamp == 0:
                camera_dict[cam_name] = cam_info # needed for video camera
            cam_infos.append(cam_info)
    return cam_infos, camera_dict

def readBricsSceneInfo(path, num_pts=200_000, white_background=True, start_t=0, num_t=1, init='hull', create_video_cams=True, load_image_on_the_fly = False):
    print("Reading Brics Info")
    train_cam_infos, train_camera_dict = readBrics(path, split='train', white_background=white_background, start_t=start_t, num_t=num_t, load_image_on_the_fly = load_image_on_the_fly)
    test_cam_infos, _ = readBrics(path, split='test', white_background=white_background, start_t=start_t, num_t=num_t, load_image_on_the_fly = load_image_on_the_fly)

    # init points
    if init == 'hull':
        first_frame_cameras = [_cam for _cam in train_cam_infos if int(_cam.fid*num_t)%100 == 0]
        aabb = -3.0, 3.0
        grid_resolution = 128
        grid = np.linspace(aabb[0], aabb[1], grid_resolution)
        grid = np.meshgrid(grid, grid, grid)
        grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

        # project grid locations to the image plane
        grid = torch.from_numpy(np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1)).float() # n_pts, 4
        # grid_mask = np.ones_like(grid_loc[:, 0], dtype=bool)
        grid_counter = np.ones_like(grid_loc[:, 0], dtype=int)
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        for cam in first_frame_cameras:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)

            if not load_image_on_the_fly:
                H, W = cam.image.size[1], cam.image.size[0]
            else:
                img, mask = load_img(cam.image_path, white_background = white_background)
                H, W = img.size[1], img.size[0]

            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY, K=cam.K, img_h=H, img_w=W).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            # xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = grid @ full_proj_transform # (full_proj_transform @ xyzh.T).T
            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            # H, W = cam.image.size[1], cam.image.size[0]
            uv = ndc2Pix(uv, np.array([W, H]))
            uv = np.round(uv.numpy()).astype(int)

            valid_inds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) 
            # _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not load_image_on_the_fly:
                cam_mask = np.array(cam.mask) # H,W,1
            else:
                cam_mask = np.array(mask) # H,W,1
            # _pix_mask[_pix_mask] = cam_mask[uv[valid_inds][:, 1], uv[valid_inds][:, 0]].reshape(-1) > 0

            _m = cam_mask[uv[valid_inds][:, 1], uv[valid_inds][:, 0]].reshape(-1) > 0
            # grid_mask[valid_inds] = grid_mask[valid_inds] & _m
            grid_counter[valid_inds] = grid_counter[valid_inds] + _m
            print('grid_counter=', np.mean(grid_counter))

            if True:
                cam_img = np.array(cam.image if not load_image_on_the_fly else img).copy()
                red_uv = uv[valid_inds][_m > 0]
                cam_img[red_uv[:, 1], red_uv[:, 0]] = np.array([255, 0, 0])
                # save cam_img
                imageio.imsave(f'./cam_img.png', cam_img)
                # breakpoint()

        grid_mask = grid_counter > 15 # at least 10 cameras should see the point
        xyz = grid[:, :3].numpy()[grid_mask]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
        ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")

    else:
        raise NotImplementedError

    # sub sample points if needed
    if xyz.shape[0] > num_pts:
        xyz = xyz[np.random.choice(xyz.shape[0], num_pts, replace=False)]
    colors = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    storePly(ply_path, xyz, colors)

    # create visualization cameras
    video_cameras = []
    if create_video_cams:
        vis_C2W = []
        vis_cam_order = ['cam01', 'cam04', 'cam09', 'cam15', 'cam23', 'cam28', 'cam32', 'cam34', 'cam35', 'cam36', 'cam37'] + ['cam01', 'cam04']
        cam_id_order = [train_camera_dict[vis_cam_id] for vis_cam_id in vis_cam_order]
        for cam in cam_id_order:
            Rt = np.eye(4)
            Rt[:3, :3] = cam.R
            Rt[:3, 3] = cam.T
            vis_C2W.append(np.linalg.inv(Rt))
        vis_C2W = np.stack(vis_C2W)[:, :3, :4]
        # interpolate between cameras
        visualization_poses = generate_interpolated_path(vis_C2W, 50, spline_degree=3, smoothness=0.0, rot_weight=0.01)
        video_cam_centers = []
        # timesteps = list(range(start_t, start_t+num_t))
        timesteps = list(range(0, num_t))
        timesteps_rev = timesteps + timesteps[::-1]
        for _idx, _pose in enumerate(visualization_poses):
            Rt = np.eye(4)
            Rt[:3, :4] = _pose[:3, :4]
            Rt = np.linalg.inv(Rt)
            R = Rt[:3, :3]
            T = Rt[:3, 3]
            video_cameras.append(CameraInfo(
                    uid=_idx,
                    fid=timesteps_rev[_idx % len(timesteps_rev)]/float(num_t), # iterate over the time cameras
                    R=R, T=T,
                    FovY=train_cam_infos[0].FovY, FovX=train_cam_infos[0].FovX,
                    image=None, image_path=None, image_name=f"{_idx:05}", 
                    width=train_cam_infos[0].width, height=train_cam_infos[0].height, white_background = white_background
                    # width=train_cam_infos[0].image.size[0], height=train_cam_infos[0].image.size[1],
            ))
            video_cam_centers.append(_pose[:3, 3])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cameras,
                           nerf_normalization=getNerfppNorm(train_cam_infos),
                           ply_path=ply_path
                           )
    return scene_info


def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0].image
    # if split == "train":
    for idx in tqdm(range(len(dataset))):
        image_path = None
        image_name = f"{idx}"
        time = dataset.image_times[idx]
        # matrix = np.linalg.inv(np.array(pose))
        R,T = dataset.load_pose(idx)
        if hasattr(dataset, "focal"):
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
        elif hasattr(dataset, "FovX") and hasattr(dataset, "FovY"):
            FovX = dataset.FovX
            FovY = dataset.FovY
        else:
            raise ValueError("No focal length or FovX and FovY found")
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            fid = time, mask=None, white_background = True, K=None)) #TODO

    return cameras

def readEgooodSceneInfo(datadir, use_bg_points, load_image_on_the_fly = False, start_t=2520, num_t=3840-2520):
    from scene.egoood_dataset import EgoOodDataset
    train_dataset = EgoOodDataset(datadir, start_frame = start_t, end_frame = start_t+ num_t, split='train', load_image_on_the_fly = load_image_on_the_fly)
    test_dataset = EgoOodDataset(datadir, start_frame = start_t, end_frame = start_t+ num_t, split='test', load_image_on_the_fly = load_image_on_the_fly)

    # train_cam_infos = format_infos(train_dataset,"train")
    nerf_normalization = getNerfppNorm(train_dataset) #I don't think it will be used
    # nerf_normalization = None
    ply_path  = os.path.join(datadir, 'pcd/egoseq/start-2520_dense_downsample.ply')
    pcd = fetchPly(ply_path)
    print("Origin points", pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_dataset,
                            test_cameras=test_dataset,
                            video_cameras=None, #TODO
                            ply_path=ply_path,
                            nerf_normalization=nerf_normalization
                            )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "Brics": readBricsSceneInfo,
    "EgoOod": readEgooodSceneInfo,
}
