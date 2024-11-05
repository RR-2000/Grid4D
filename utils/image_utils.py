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

import torch
from PIL import Image
import numpy as np


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_img(img_path, downsample: int = 1, white_background: bool = False):
    image = Image.open(img_path)
    
    if downsample > 1:
        image = image.resize((image.size[0]//downsample, image.size[1]//downsample), Image.ANTIALIAS)
    im_data = np.array(image.convert("RGBA"))

    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

    norm_data = im_data / 255.0
    mask = norm_data[..., 3:4]

    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

    return image, mask