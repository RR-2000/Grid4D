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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.dataset import FourDGSdataset, EgoLoading
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.num_gaussians = len(gaussians)
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "frames_1")):
            print("Found frames_1 directory, assuming Brics data set!")
            scene_info = sceneLoadTypeCallbacks["Brics"](
                args.source_path,
                white_background=args.white_background,
                start_t=args.start_t,
                num_t=args.num_t,
                load_image_on_the_fly = args.load_image_on_the_fly,
            )
            dataset_type="brics"
        elif os.path.exists(os.path.join(args.source_path, "pcd")):
            scene_info = sceneLoadTypeCallbacks["EgoOod"](args.source_path, args.white_background)
            dataset_type="EgoOod"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle and type(scene_info.train_cameras) == list:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Camera radius:", self.cameras_extent, "Camera center:", scene_info.nerf_normalization['translate'])
        print("Point bound:", np.max(scene_info.point_cloud.points, axis=0), "<->", np.min(scene_info.point_cloud.points, axis=0))

        for resolution_scale in resolution_scales:
            if dataset_type == "EgoOod":

                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = EgoLoading(scene_info.train_cameras, args, dataset_type, resolution_scale)

                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = EgoLoading(scene_info.test_cameras, args, dataset_type, resolution_scale)

                print("Loading Video Cameras")
                self.video_cameras[resolution_scale] = EgoLoading(scene_info.video_cameras, args, dataset_type, resolution_scale)

                continue
            
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = FourDGSdataset(scene_info.train_cameras, args, dataset_type, resolution_scale) if args.load_image_on_the_fly else cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = FourDGSdataset(scene_info.test_cameras, args, dataset_type, resolution_scale) if args.load_image_on_the_fly else cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
            print("Loading Video Cameras")
            self.video_cameras[resolution_scale] = FourDGSdataset(scene_info.video_cameras, args, dataset_type, resolution_scale) if args.load_image_on_the_fly else cameraList_from_camInfos(scene_info.video_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            for i in range(self.num_gaussians):
                self.gaussians[i].load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    f"point_cloud_{i}.ply"),
                                        og_number_points=len(scene_info.point_cloud.points))
        else:
            for i in range(self.num_gaussians):
                self.gaussians[i].create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        print("Train camera:", len(self.getTrainCameras()), "Test camera:", len(self.getTestCameras()))

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            os.makedirs(point_cloud_path, exist_ok=True)
            with open(os.path.join(point_cloud_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        for i in range(self.num_gaussians):
            self.gaussians[i].save_ply(os.path.join(point_cloud_path, f"point_cloud_{i}.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVideoCameras(self, scale=1.0):
        return self.video_cameras[scale]