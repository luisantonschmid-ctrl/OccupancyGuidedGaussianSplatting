#from renderplytest import ply_path
from AnySplat.src.model.types import Gaussians
from AnySplat.src.model.encoder.common.gaussians import build_covariance
from math import sqrt
import os
from scipy.spatial.transform import Rotation as R
from AnySplat.src.model.model.anysplat import AnySplat
from AnySplat.src.model.ply_export import export_ply
from optimizer_mesh import PoseOptimizer
import sys
from plyfile import PlyData, PlyElement
import trimesh
import numpy as np
#from kittiseq10_config import Config
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation
from gsplat import rasterization
import torch
from PIL import Image
import torchvision
import time
from scipy.interpolate import RBFInterpolator
import math
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import yaml
import importlib.util
import json
#from skywater_seg import load_model, segment_skywater
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import yaml
from scipy.ndimage import distance_transform_edt
import skimage
import numpy as np
import cv2
@dataclass
class Config:
    pose_path: str
    image_path: str
    calib_path: str
    img_w: int
    img_h: int
    vol_radius: int
    submap_folder:str
    output_dir: str

def replace_white_pixels(img1, img2, threshold=240):
    """
    Replace white/near-white pixels in img1 with pixels from img2.

    Images: NumPy arrays with shape (H, W, 3), values [0, 255].
    """
    # Identify pixels where all three channels exceed the threshold
    mask = np.all(img1 >= threshold, axis=2)

    # Replace matching pixels
    result = img1.copy()
    result[mask] = img2[mask]

    return result
def replace_black_pixels(img1, img2, threshold=5):
    """
    Replace black or nearly black pixels in img1 with corresponding pixels from img2.

    Args:
        img1: First image, shape (H, W, 3).
        img2: Second image, shape (H, W, 3).
        threshold: Pixels with all channels <= threshold are replaced.

    Returns:
        Modified copy of img1.
    """
    img1 = np.array(img1).copy()
    img2 = np.asarray(img2)

    # Identify black and nearly black pixels
    mask = np.all(img1 <= threshold, axis=2)

    # Replace them with corresponding pixels from img2
    img1[mask] = img2[mask]

    return img1

import cv2
import numpy as np


def mask_black_pixels_far_from_nonblack(image, x=10, threshold=5):
    """
    Masks black pixels that have no non-black pixel within distance x.

    Args:
        image: Input image, shape (H, W, 3).
        x: Minimum distance from any non-black pixel.
        threshold: Maximum intensity considered black.

    Returns:
        Integer mask of shape (H, W, 3).
        1 = qualifying black pixel
        0 = otherwise
    """
    # Identify black pixels (all color channels must be dark)
    black = np.all(image <= threshold, axis=2)

    # Distance transform measures distance to the nearest non-black pixel.
    # OpenCV measures distance to the nearest zero pixel.
    distance_input = black.astype(np.uint8)

    distance = cv2.distanceTransform(
        distance_input,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    # Keep black pixels whose nearest non-black pixel is at least x away
    mask = black & (distance >= x)

    # Convert to 3-channel integer mask
    mask = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8)

    return np.array(~np.array(mask,dtype=bool),dtype=np.uint8)
def render_scene(config):

    calib = pd.read_csv(config.calib_path, delimiter=' ', header=None, index_col=0)
    p2 = np.array(calib.loc['P2:']).reshape((3, 4))
    K, r2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(p2)

    poses = []
    with open(Path(config.pose_path)) as f:
        for line in f:
            vals = np.fromstring(line.strip(), sep=' ')
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = vals.reshape(3, 4)
            poses.append(T)
    poses=np.stack(poses)
    imgfolder = Path(config.image_path)
    imgnames=sorted(imgfolder.glob("*.png"))
    renpath = config.output_dir + '/rendersb_r' + str(config.vol_radius)
    meshpath = config.output_dir + '/renders_mesh'
    renfolder = Path(renpath)
    rennames=sorted(renfolder.glob("*.png"))
    meshfolder = Path(meshpath)
    meshnames=sorted(meshfolder.glob("*.png"))
    psnrs=[]
    ssims=[]
    psnrs_m=[]
    ssims_m=[]
    psnrs_o=[]
    ssims_o=[]
    plot=False
    #plot=True
    for i in range(len(imgnames)):
        img = Image.open(imgnames[i])
        if img.mode == 'RGBA':
            # Convert RGBA to RGB by removing alpha channel
            img = img.convert('RGB')
        img=np.asarray(img)
        ren = Image.open(rennames[i])
        if ren.mode == 'RGBA':
            # Convert RGBA to RGB by removing alpha channel
            ren = ren.convert('RGB')
        ren = np.asarray(ren).copy()
        #ren[np.all(ren == 255, axis=2)] = 0

        mesh = Image.open(meshnames[i])
        if mesh.mode == 'RGBA':
            # Convert RGBA to RGB by removing alpha channel
            mesh = mesh.convert('RGB')
        mesh = np.asarray(mesh).copy()
        #mesh[np.all(mesh == 0, axis=2)] = 255
        mask =mask_black_pixels_far_from_nonblack(mesh)
        #mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        #mesh[np.all(mesh == 0, axis=2)] = 255
        #overlay=replace_white_pixels(ren,mesh)
        overlay=replace_black_pixels(ren,mesh)
        #mesh[np.all(mesh == 0, axis=2)] = 255

        '''img = img.copy()
        ren = ren.copy()
        mesh = mesh.copy()
        overlay = overlay.copy()
        img[~mask] = 0
        ren[~mask] = 0
        mesh[~mask] = 0
        overlay[~mask] = 0'''
        if plot:
            f, axarr = plt.subplots(4, 1)
            axarr[0].imshow(img*mask)
            axarr[1].imshow(ren*mask)
            axarr[2].imshow(mesh*mask)
            axarr[3].imshow(overlay*mask)

            plt.show()
        print('Metrics for image: ',i)
        psnr = skimage.metrics.peak_signal_noise_ratio(img*mask,mesh*mask, data_range=255)
        psnrs_m.append(psnr)
        print('PSNR mesh: ',psnr)
        ssim = skimage.metrics.structural_similarity(img*mask,mesh*mask, data_range=255,channel_axis=2)
        ssims_m.append(ssim)
        print('SSIM mesh: ',ssim)
        psnr = skimage.metrics.peak_signal_noise_ratio(img*mask,ren*mask, data_range=255)
        psnrs.append(psnr)
        print('PSNR: ',psnr)
        ssim = skimage.metrics.structural_similarity(img*mask,ren*mask, data_range=255,channel_axis=2)
        ssims.append(ssim)
        print('SSIM: ',ssim)
        psnr = skimage.metrics.peak_signal_noise_ratio(img*mask,overlay*mask, data_range=255)
        psnrs_o.append(psnr)
        print('PSNR overlay: ',psnr)
        ssim = skimage.metrics.structural_similarity(img*mask,overlay*mask, data_range=255,channel_axis=2)
        ssims_o.append(ssim)
        print('SSIM overlay: ',ssim)
    evalfolder='/home/luis/Desktop/Evaluations/'
    psnrpath =evalfolder+'kitti_seq10w_mesh_psnr_masked.csv'
    ssimpath =evalfolder+'kitti_seq10w_mesh_ssim_masked.csv'

    psnrpath = Path(psnrpath)
    psnrpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(psnrpath, np.array(psnrs_m), delimiter=",")
    ssimpath = Path(ssimpath)
    ssimpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(ssimpath, np.array(ssims_m), delimiter=",")
    psnrpath =evalfolder+'kitti_seq10w_psnr_masked.csv'
    ssimpath =evalfolder+'kitti_seq10w_ssim_masked.csv'

    psnrpath = Path(psnrpath)
    psnrpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(psnrpath, np.array(psnrs), delimiter=",")
    ssimpath = Path(ssimpath)
    ssimpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(ssimpath, np.array(ssims), delimiter=",")
    psnrpath =evalfolder+'kitti_seq10w_comb_psnr_masked.csv'
    ssimpath =evalfolder+'kitti_seq10w_comb_ssim_masked.csv'

    psnrpath = Path(psnrpath)
    psnrpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(psnrpath, np.array(psnrs_o), delimiter=",")
    ssimpath = Path(ssimpath)
    ssimpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(ssimpath, np.array(ssims_o), delimiter=",")

    return


def load_config(file_path: str = "config.yaml") -> Config:
    """Load a YAML file into a Config object."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}

    return Config(**data)
def main():
    configpath = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/kittiseq10_config.yaml"
    config = load_config(configpath)
    #multisubmap(config)
    render_scene(config)
    return


if __name__ == '__main__':
    sys.exit(main())