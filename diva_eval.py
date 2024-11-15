## Reference: https://github.com/brown-ivl/DiVa360/blob/main/utils/benchmark.py

import os
import glob
# import json
import numpy as np
import cv2 as cv
import torch
from argparse import ArgumentParser
from skimage.metrics import structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from scene.dataset import evalDataset
from torch.utils.data import DataLoader


def lpips_loss(image_pred, image_gt, lpips_net, device = torch.device('cuda')):
    image_pred = torch.from_numpy(image_pred).float().to(device)
    image_gt = torch.from_numpy(image_gt).float().to(device)
    image_pred = image_pred.unsqueeze(0).permute(0,3,1,2)
    image_gt = image_gt.unsqueeze(0).permute(0,3,1,2)
    
    # Normalizing the images to [-1, 1]
    image_pred = image_pred * 2 - 1
    image_gt = image_gt * 2 - 1
    return lpips_net(image_pred, image_gt).detach().cpu().numpy()


def load_image(path):
    try:
        image = cv.cvtColor(cv.imread(path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)
    except:
        raise FileNotFoundError(f"{path} does not exist")

    return image

def load_image_pair(path):
    
    gt_path = path
    pred_path = path.replace("gt", "renders")

    gt =  load_image(gt_path)
    pred = load_image(pred_path)

    return gt, pred



def eval_set(model_path, device = torch.device('cuda'), wh_bg=False):

    # Each experiment is expected to be in the form: model/checkpoint/gt/camXX/%05d.png
    # gt_paths = glob.glob(model_path+'/gt/*/*.png')
    dataset = evalDataset(model_path)
    dataloader = DataLoader(dataset, num_workers = 32, batch_size = 1)

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    count = 0
    lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    for view_pair in tqdm(iter(dataloader), desc = f"Model: {model_path.split('/')[-3]} Split: {model_path.split('/')[-2]} Checkpoint: {model_path.split('/')[-1]}", position = 1):

        # gt, pred = load_image_pair(view_path)
        gt = view_pair[0][0].cpu().numpy()
        pred = view_pair[1][0].cpu().numpy()

        gt = gt.astype(np.float32)
        gt /= 255.
        bg_color = 1.0 if wh_bg else 0.0
        gt = gt[..., :3]*gt[..., 3:4] + (1.-gt[..., 3:4])*bg_color
        gt = (gt*255).astype(np.uint8)
        pred = pred.astype(np.float32)
        pred /= 255.
        pred = pred[..., :3]*pred[..., 3:4] + (1.-pred[..., 3:4])*bg_color
        pred = (pred*255).astype(np.uint8)

        psnr = cv.PSNR(gt, pred)
        ssim = structural_similarity(gt, pred, channel_axis=2)
        gt = gt.astype(np.float32)
        gt /= 255.
        pred = pred.astype(np.float32)
        pred /= 255.
        lpips = float(lpips_loss(pred, gt, lpips_net, device))
        count += 1

        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips

    print(f"Model: {model_path.split('/')[-3]} Split: {model_path.split('/')[-2]} Checkpoint: {model_path.split('/')[-1]}")
    print("PSNR", avg_psnr/count)
    print("SSIM", avg_ssim/count)
    print("LPIPS", avg_lpips/count)
    print()


def eval_sets(exp_dir, device = torch.device('cuda'), wh_bg=False, skip_train = False, skip_test = False):

    
    test_dir = glob.glob(exp_dir + "/*/test/*")
    train_dir = glob.glob(exp_dir + "/*/train/*")

    model_folders = []

    if not skip_train:
        model_folders.extend(train_dir)
    
    if not skip_test:
        model_folders.extend(test_dir)

    model_folders = sorted(model_folders)
    models = []
    for folder in model_folders:
        if len(glob.glob(folder + "/gt/*.png")) > 0 and len(glob.glob(folder + "/renders/*.png")) > 0:
            models.append(folder)

    for model_path in tqdm(models, desc = "Overall", position = 0):
            eval_set(model_path = model_path, device = device, wh_bg = wh_bg)
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Folder with all models dirs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wh_bg", action='store_true')
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument("--skip_test", action='store_true')
    args = parser.parse_args()
        
    eval_sets(args.exp_dir, device = args.device, wh_bg = args.wh_bg, skip_test= args.skip_test, skip_train= args.skip_train)