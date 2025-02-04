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
import torch
from random import randint, choice
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
from scene.dataset_readers import CameraInfo
from utils.camera_utils import loadCam
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, quaternion_to_matrix
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, merge_config
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer, args = prepare_output_and_logger(dataset)
    
    deform = DeformModel(
        grid_args=dataset.grid_args, 
        net_args=dataset.network_args,
        spatial_downsample_ratio=opt.spatial_downsample_ratio,
        spatial_perturb_range=opt.spatial_perturb_range, 
        temporal_downsample_ratio=opt.temporal_downsample_ratio,
        temporal_perturb_range=opt.temporal_perturb_range, 
        scale_xyz=dataset.scale_xyz,
        reg_spatial_able=opt.lambda_spatial_tv > 0.0,
        reg_temporal_able=opt.lambda_temporal_tv > 0.0,
    )
    deform.train_setting(opt)

    gaussians = []
    for i in range(dataset.num_gaussians):
        gaussians.append(GaussianModel(dataset.sh_degree))
    
    scene = Scene(dataset, gaussians)

    for i in range(dataset.num_gaussians):
        gaussians[i].training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            for i in range(dataset.num_gaussians):
                gaussians[i].oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            if dataset.load_image_on_the_fly:
                viewpoint_stack = scene.getTrainCameras()
                sampler = RandomSampler(viewpoint_stack, replacement=True, num_samples=opt.iterations)
                viewpoint_stack_loader = iter(DataLoader(viewpoint_stack, sampler=sampler, batch_size=1, num_workers = opt.num_workers, collate_fn=list))
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
        
        if not dataset.load_image_on_the_fly:

            if opt.data_sample == 'random':
                viewpoint_cam = choice(scene.getTrainCameras())
            elif opt.data_sample == 'order':
                viewpoint_cam = viewpoint_stack.pop(0)
            elif opt.data_sample == 'stack':
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        else:
            viewpoint_cam = next(viewpoint_stack_loader)[0]
            if isinstance(viewpoint, CameraInfo):
                viewpoint_cam = loadCam(viewpoint_stack.args, 0, viewpoint_cam, viewpoint_stack.resolution_scale)


        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        gaussian_id = int(fid*dataset.num_gaussians)

        # deformation and regularization
        reg = 0.0
        if iteration < opt.warm_up:
            d_rotation, d_scaling = 0.0, 0.0
            d_xyz = 0.0
        else:
            N = gaussians[gaussian_id].get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            xyz = gaussians[gaussian_id].get_xyz.detach()
            deform_pkgs = deform.step(xyz, time_input)
            d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

            if opt.lambda_spatial_tv > 0.0 and (not opt.reg_after_densify or iteration > opt.densify_until_iter):
                reg += torch.mean(deform_pkgs['reg_spatial']) * opt.lambda_spatial_tv
            if opt.lambda_temporal_tv > 0.0 and (not opt.reg_after_densify or iteration > opt.densify_until_iter):
                reg += torch.mean(deform_pkgs['reg_temporal']) * opt.lambda_temporal_tv


        # Render
        render_pkg_re = render(viewpoint_cam, gaussians[gaussian_id], pipe, background, d_xyz, d_rotation, d_scaling)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + reg
        loss.backward()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "pts": len(gaussians[gaussian_id].get_xyz),
                    "reg": f"{reg:.{5}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians[gaussian_id].max_radii2D[visibility_filter] = torch.max(gaussians[gaussian_id].max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly)

            if iteration in testing_iterations:
                if cur_psnr.item() >= best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration
                    scene.save(iteration, True)
                    deform.save_weights(args.model_path, iteration, True)
                    print("Best: {} PSNR: {}".format(best_iteration, best_psnr))

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians[gaussian_id].add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians[gaussian_id].densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.disable_ws_prune)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians[gaussian_id].reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians[gaussian_id].optimizer.step()
                gaussians[gaussian_id].update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians[gaussian_id].optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in
                                           range(5, 100, 5)]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 100, 5)]})

        for config in validation_configs:
            render_path = f'bike/{config["name"]}'
            os.makedirs(render_path, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = []
                psnr_test = []
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    if isinstance(viewpoint, CameraInfo):
                        viewpoint = loadCam(scene.getTestCameras().args, 0, viewpoint, scene.getTestCameras().resolution_scale)
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    gaussian_id = int(fid*len(scene.gaussians))
                    xyz = scene.gaussians[gaussian_id].get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    deform_pkgs = deform.step(xyz.detach(), time_input)
                    d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians[gaussian_id], *renderArgs, d_xyz, d_rotation, d_scaling)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                    torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name + ".png"))
                    l1_test.append(l1_loss(image, gt_image).mean().item())
                    psnr_test.append(psnr(image, gt_image).mean().item())

                l1_test = np.mean(l1_test)
                psnr_test = np.mean(psnr_test)
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians[0].get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians[0].get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                       default=[5000] + list(range(10000, 50001, 5000))) #default=[5000, 6000, 7_000] + list(range(10000, 50001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 5000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    if args.conf is not None and os.path.exists(args.conf):
        print("Find Config:", args.conf)
        args = merge_config(args, args.conf)
    else:
        print("[WARNING] Using default config.")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.data_device = "cuda:0" if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)

    if not args.quiet:
        print(vars(args))

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
