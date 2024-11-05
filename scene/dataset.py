from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
import cv2 as cv
import glob
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        resolution_scale = 1
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.resolution_scale = resolution_scale
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None

                return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu',
                              time=time,
                              mask=mask)
            except:
                caminfo = self.dataset[index]
                # image = caminfo.image
                # R = caminfo.R
                # T = caminfo.T
                # FovX = caminfo.FovX
                # FovY = caminfo.FovY
                # time = caminfo.time
                # K = caminfo.K
    
                # mask = caminfo.mask
                return loadCam(self.args, index, caminfo, self.resolution_scale)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)


class evalDataset(Dataset):
    def __init__(
        self, model_path
    ):
        self.model_path = model_path
        self.gt_paths = glob.glob(model_path+'/gt/*/*.png')
        
    def __getitem__(self, index):

        gt_path = self.gt_paths[index]

        return self.load_image_pair(gt_path)

    def __len__(self):
        
        return len(self.gt_paths)

    def load_image(self, path):
        try:
            image = cv.cvtColor(cv.imread(path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)
        except:
            raise FileNotFoundError(f"{path} does not exist")

        return image

    def load_image_pair(self, path):
    
        gt_path = path
        pred_path = path.replace("gt", "renders")

        gt =  self.load_image(gt_path)
        pred = self.load_image(pred_path)

        return gt, pred
