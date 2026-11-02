from src.model.types import Gaussians

from pathlib import Path
import torch
import os
import sys
import pandas as pd
from AnySplat.src.misc.image_io import save_interpolated_video
from AnySplat.src.model.model.anysplat import AnySplat
#from src.utils.image import process_image
from src.model.ply_export import export_ply
import glob
from os import listdir
from os.path import isfile, join
import cv2

import numpy as np
import matplotlib.pyplot as plt
from gsplat import rasterization
from plyfile import PlyData, PlyElement
import json
import sys
import trimesh
from networkx.algorithms.distance_measures import radius
from plyfile import PlyData, PlyElement
import trimesh
import numpy as np
import os
from kittiseq00_config import Config
#from scene.gaussian_model import GaussianModel, BasicPointCloud
from pyntcloud import PyntCloud
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import math
import glob
import random
from scipy.spatial.transform import Rotation
from gsplat import rasterization
import torch
import json
from PIL import Image
import torchvision
import time

NEAR_CLIP = 0.5  # metres — discard points closer than this
FAR_CLIP = 80.0  # metres — discard points farther than this
POINT_SIZE = 2  # blob radius in pixels (1 = single pixel)
OVERLAY_ALPHA = 0.6  # mesh weight in the blended overlay image


def process_image_helper(np_image):
    img = Image.fromarray(np_image, 'RGB')
    #img = Image.open(img_path)
    if img.mode == 'RGBA':
        # Convert RGBA to RGB by removing alpha channel
        img = img.convert('RGB')
    # Resize to maintain aspect ratio and then center crop to 448x448
    width, height = img.size
    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))
    img = img.resize((new_width, new_height))

    # Center crop
    left = (new_width - 448) // 2
    top = (new_height - 448) // 2
    right = left + 448
    bottom = top + 448
    img = img.crop((left, top, right, bottom))
    #img.show()
    #time.sleep(10)
    img_tensor = torchvision.transforms.ToTensor()(img) * 2.0 - 1.0  # [-1, 1]
    return img_tensor

def render_splatt(ply,K,cam):
    v = ply["vertex"].data

    means = np.stack([v["x"], v["y"], v["z"]], axis=1)

    scales = np.stack([
        v["scale_0"],
        v["scale_1"],
        v["scale_2"],
    ], axis=1)

    quats = np.stack([
        v["rot_0"],
        v["rot_1"],
        v["rot_2"],
        v["rot_3"],
    ], axis=1)

    opacity = v["opacity"][:, None]

    # Degree-0 SH (RGB)
    colors = np.stack([
        v["f_dc_0"],
        v["f_dc_1"],
        v["f_dc_2"],
    ], axis=1)

    device = "cuda"

    means = torch.tensor(means, dtype=torch.float32, device=device)
    scales = torch.exp(torch.tensor(scales, dtype=torch.float32, device=device))
    quats = torch.tensor(quats, dtype=torch.float32, device=device)
    opacity = torch.sigmoid(torch.tensor(opacity, dtype=torch.float32, device=device))
    colors = torch.tensor(colors, dtype=torch.float32, device=device)



    K = torch.tensor(K, dtype=torch.float32, device="cuda")[None]

    # World -> Camera transform
    viewmat = torch.tensor(cam, dtype=torch.float32, device=device)[None]

    rgb, alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=255,
        height=255,
    )
    image = rgb[0].cpu().numpy()
    print(image)

    return image

def get_pixel_3d_coord(u, v, gaussians, H, W, view_idx=0, batch_idx=0):
    """
    Retrieves the 3D world coordinate directly from AnySplat's pixel-wise Gaussians.
    u = column index (horizontal coordinate, 0 to W-1)
    v = row index    (vertical coordinate, 0 to H-1)
    """
    # Calculate the flattened sequence index for the specific view and pixel
    # AnySplat structures its token sequences row-major: view -> row (v) -> col (u)
    pixel_flat_idx = (view_idx * H * W) + (v * W) + u

    # Query directly from the Gaussian means tensor [Batch, Total_Pixels, 3]
    world_xyz = gaussians.means[batch_idx, pixel_flat_idx]

    return world_xyz.tolist()

class Submap:
    def __init__(self,config,frame_start,frame_end):
        self.config = config
        self.calib=pd.read_csv(config.calib_path, delimiter=' ', header=None, index_col=0)
        p2 = np.array(self.calib.loc['P2:']).reshape((3, 4))
        K, r2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(p2)
        self.K = K
        self.R = r2
        self.T = t2
        self.img_h =config.img_h
        self.img_w = config.img_w
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.vol_radius = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.poses
        #self.mesh
        #self.pcd
        #self.centers
        #self.images
        #self.inverted_poses
        #self.im_u
        #self.im_v
        #self.inside_mask

        #self.crop_dict
        #self.center_shiftdict

        #self.model

    def init_model(self):
        self.model = AnySplat.from_pretrained("lhjiang/anysplat")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return
    def init_geometry(self,mesh_path):
        self.poses=self.load_poses(Path(self.config.pose_path))
        self.mesh = trimesh.load_mesh(mesh_path)
        self.pcd = self.mesh.triangles_center.astype(np.float32)
        self.centers = self.fetch_centroids(self.vol_radius)
        self.images = self.fetch_images()
        self.inverted_poses = self.invert_poses()
        self.im_u = np.empty((0, len(self.centers)))
        self.im_v = np.empty((0,len(self.centers)))
        self.inside_mask = np.empty((0, len(self.centers)))

        for i in range(len(self.images)):
            u_centers, v_centers, in_img = self.project_centers(i)
            self.im_u = np.vstack((self.im_u, u_centers))
            self.im_v = np.vstack((self.im_v, v_centers))
            self.inside_mask = np.vstack((self.inside_mask, in_img))
        return

    def init_model_inputs(self):
        self.crop_dict = {}
        self.center_shiftdict = {}
        for j in range(len(self.centers)):
            for k in range(self.frame_end):
                if self.inside_mask[k][j]:
                    #print(self.im_u[k][j], self.im_v[k][j])
                    crop, centershift = self.extract_window_no_padding(self.images[k], int(self.im_u[k][j]),
                                                                         int(self.im_v[k][j]), plot=False)
                    self.center_shiftdict[str("center" + str(j) + "image" + str(k))] = {"x_shift": str(centershift[0]),
                                                                                   "y_shift": str(centershift[1])}
                    self.crop_dict[str("center" + str(j) + "image" + str(k))] = {"crop": crop}
                    #cv2.imwrite('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/mini-splatting2/output/cropped2m/center' + str(j) + 'image' + str(k) + '.png', crop)
        return
    def render_splat(self,gaussians,pred_all_extrinsic,pred_all_intrinsic,h=224,w=224):

        interpolated_output = self.model.decoder.forward(
            gaussians,
            pred_all_extrinsic,
            pred_all_intrinsic.float(),
            torch.ones(1, 2, device=pred_all_extrinsic.device) * 0.1,
            torch.ones(1, 2, device=pred_all_extrinsic.device) * 100,
            (h, w),
        )

        video = interpolated_output.color[0].clip(min=0, max=1)

        video = video.cpu().detach().numpy()  # (N, C, H, W)
        video = np.transpose(video, (0, 2, 3, 1))  # (N, H, W, C)

        # Scale to [0, 255] and convert to uint8
        video = (video * 255).astype(np.uint8)
        googoldcounter = 0
        # ply = PlyData.read("/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/mini-splatting2/output/gaussians2m2pics/gaussians"+str(i)+".ply")
        # plt.imshow((np.clip(render1, 0, -1) * -255).astype(np.uint8), interpolation='nearest')
        # plt.show()
        for frame in video:
            plt.imshow(frame, interpolation='nearest')
            plt.show()
            time.sleep(5)
        #    cv2.imwrite(crop_paths[googoldcounter].replace('cropped2m','cropped2mreconstructed'),frame)
        #    googoldcounter+=1
        return
    def pipeline_foward(self):
        for i in range(len(self.centers)):

            crop_paths=list(filter(lambda x: x.startswith("center"+str(i)+"image"), sorted(list(self.center_shiftdict.keys()))))
            crop_paths=[crop_paths[0],crop_paths[-1]]
            #print(crop_paths)
            imagenr1 = int(crop_paths[0].removeprefix("center"+str(i)+"image"))
            imagenr2 = int(crop_paths[1].removeprefix("center"+str(i)+"image"))
            images = [process_image_helper(self.crop_dict[crop_paths[0]]["crop"]),process_image_helper(self.crop_dict[crop_paths[-1]]["crop"])]
            images = torch.stack(images, dim=0).unsqueeze(0).to(self.device)  # [1, K, 3, 448, 448]
            b, v, _, h, w = images.shape

            target_u1, target_v1 = 112-int(self.center_shiftdict[crop_paths[0]]['x_shift']), 112-int(self.center_shiftdict[crop_paths[0]]['y_shift'])
            target_u2, target_v2 = 112-int(self.center_shiftdict[crop_paths[1]]['x_shift']), 112-int(self.center_shiftdict[crop_paths[1]]['y_shift'])
            plot=True
            if plot:
                image1=self.crop_dict[crop_paths[0]]["crop"].copy()
                image2=self.crop_dict[crop_paths[-1]]["crop"].copy()
                image1[target_v1][target_u1] = [255, 0, 0]
                image2[target_v2][target_u2] = [255, 0, 0]
                try:
                    image1[target_v1+1][target_u1] = [255, 0, 0]
                    image2[target_v2+1][target_u2] = [255, 0, 0]
                    image1[target_v1-1][target_u1] = [255, 0, 0]
                    image2[target_v2-1][target_u2] = [255, 0, 0]
                    image1[target_v1][target_u1+1] = [255, 0, 0]
                    image2[target_v2][target_u2+1] = [255, 0, 0]
                    image1[target_v1][target_u1-1] = [255, 0, 0]
                    image2[target_v2][target_u2-1] = [255, 0, 0]
                except:
                    print('couldnt fill all surrounding pixels')
                #f, axarr = plt.subplots(2, 1)
                #axarr[0].imshow(image1)
                #axarr[1].imshow(image2)
                cv2.imwrite('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/cropswithcenter2m/center' + str(i) +'image1'+ '.png', image1)
                cv2.imwrite('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/cropswithcenter2m/center' + str(i) +'image2'+ '.png', image2)
                #plt.show()
            distance1 = np.linalg.norm(self.centers[i] - self.poses[imagenr1][0:3,3:].reshape((3,)))
            distance2 = np.linalg.norm(self.centers[i] - self.poses[imagenr2][0:3,3:].reshape((3,)))
            print('Distance '+crop_paths[0],distance1,self.centers[i],self.poses[imagenr1][0:3,3:].reshape((3,)))
            print('Distance '+crop_paths[1],distance2,self.centers[i],self.poses[imagenr2][0:3,3:].reshape((3,)))
            #exit()
            self.model.encoder.pixeldepth1 = (target_u1,target_v1,distance1)
            #self.model.encoder.pixeldepth2 = (target_u2,target_v2,distance2)
            self.model.encoder.pixeldepth2 = (target_u2,target_v2,distance1)

            extrinsics1 = self.poses[imagenr1]
            extrinsics2 = self.poses[imagenr2]

            self.model.encoder.extrinsics1 = extrinsics1
            self.model.encoder.extrinsics2 = extrinsics2
            images = (images + 1) * 0.5
            gaussians, pred_context_pose = self.model.inference(images)

            '''xyz1 = get_pixel_3d_coord(target_u1, target_v1, gaussians, 224, 224, view_idx=0)
            xyz1rev = get_pixel_3d_coord(target_v1, target_u1, gaussians, 224, 224, view_idx=0)
            xyz2 = get_pixel_3d_coord(target_u2, target_v2, gaussians, 224, 224, view_idx=1)
            xyz2rev = get_pixel_3d_coord(target_v2, target_u2, gaussians, 224, 224, view_idx=1)
            pred_all_extrinsic = pred_context_pose['extrinsic']
            #dfex = pd.DataFrame(pred_all_extrinsic.cpu().numpy().reshape(2, 16))
            #dfex.to_csv("/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/mini-splatting2/output/gaussian2m2picsextrinsics/center"+str(i)+".csv")
            pred_all_intrinsic = pred_context_pose['intrinsic']
            #dfin = pd.DataFrame(pred_all_intrinsic.cpu().numpy().reshape(2, 9))
            #dfin.to_csv("/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/mini-splatting2/output/gaussian2m2picsintrinsics/center" + str(i) + ".csv")
            print('xyz1',xyz1)
            print('xyz1rev',xyz1rev)
            print('xyz2',xyz2)
            print('xyz2rev',xyz2rev)
            means =gaussians.means[0].cpu().detach().numpy()
            radius=2
            mask = np.sum((means - np.array([xyz1rev])) ** 2, axis=1) <= radius**2
            #means = np.concatenate((means,np.array([xyz1])),axis=0)
            means_premasked =means[mask]- np.array([xyz1rev])
            means_masked=torch.tensor(means_premasked, dtype=torch.float32, device="cuda")[None]
            scales =gaussians.scales[0]
            scales_masked = torch.squeeze(torch.tensor(scales.cpu().detach().numpy()*44, dtype=torch.float32, device="cuda")[None])
            rotations = gaussians.rotations[0]
            rotations_masked = torch.tensor(rotations.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]
            harmonics=gaussians.harmonics[0]
            harmonics_masked = torch.tensor(harmonics.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]
            opacities=gaussians.opacities[0]
            opacities_masked = torch.tensor(opacities.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]'''
            depths_scalar1=self.model.encoder.depths_scalar1
            depths_scalar2=self.model.encoder.depths_scalar2
            mean_coord1 = self.model.encoder.meancoord1
            mean_coord2 = self.model.encoder.meancoord2
            depth_pixel1 = self.model.encoder.depths_of_pixel1
            depth_pixel2 = self.model.encoder.depths_of_pixel2

            export_ply(gaussians.means[0], torch.squeeze(torch.tensor(gaussians.scales[0].cpu().detach().numpy(), dtype=torch.float32, device="cuda")[None]), gaussians.rotations[0], gaussians.harmonics[0],
                       gaussians.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled') / ("gaussians" + str(i) + ".ply"))
            means = gaussians.means[0].cpu().detach().numpy()*depths_scalar1
            radius=self.vol_radius

            #center = get_pixel_3d_coord(target_u1*2, target_v1*2, gaussians, 448, 448, view_idx=0)
            exact_center=[mean_coord1[0]*depths_scalar1, mean_coord1[1]*depths_scalar1, mean_coord1[2]*depths_scalar1]


            exactcenternp=np.array([exact_center])
            means_shifted = means -exactcenternp
            print(exactcenternp)
            #print(center)
            print(self.centers[i])
            mask=np.linalg.norm(means - exactcenternp, axis=1)<= radius
            #mask = np.sum((means - np.array([exact_center])) ** 2, axis=1) <= radius**2
            #means = np.concatenate((means,np.array([xyz1])),axis=0)
            means_masked_unshifted=torch.tensor(means[mask], dtype=torch.float32, device="cuda")[None]

            means_premasked =means[mask]- np.array([exact_center])
            means_masked=torch.tensor(means_premasked, dtype=torch.float32, device="cuda")[None]
            scales =gaussians.scales[0]
            scales_masked = torch.tensor(scales.cpu().detach().numpy()[mask]*depths_scalar1, dtype=torch.float32, device="cuda")[None]
            rotations = gaussians.rotations[0]
            rotations_masked = torch.tensor(rotations.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]
            harmonics=gaussians.harmonics[0]
            harmonics_masked = torch.tensor(harmonics.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]
            opacities=gaussians.opacities[0]
            opacities_masked = torch.tensor(opacities.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]
            covariance=gaussians.covariances[0]
            covariance_masked = torch.tensor(covariance.cpu().detach().numpy()[mask], dtype=torch.float32, device="cuda")[None]
            #export_ply(gaussians.means[0], torch.squeeze(torch.tensor(gaussians.scales[0].cpu().detach().numpy()*44, dtype=torch.float32, device="cuda")[None]), gaussians.rotations[0], gaussians.harmonics[0],
            #           gaussians.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled') / ("gaussiansmasked" + str(i) + ".ply"))
            export_ply(torch.squeeze(means_masked), torch.squeeze(scales_masked), torch.squeeze(rotations_masked), torch.squeeze(harmonics_masked),
                       torch.squeeze(opacities_masked), Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled') / ("gaussiansmasked"+str(i)+".ply"))
            pred_all_extrinsic = pred_context_pose['extrinsic']

            pred_all_intrinsic = pred_context_pose['intrinsic']

            #gaussians.means[0]=torch.squeeze(means_masked)
            #gaussians.scales[0]=torch.squeeze(scales_masked)
            #gaussians.rotations[0]=torch.squeeze(rotations_masked)
            #gaussians.harmonics[0]=torch.squeeze(harmonics_masked)
            #gaussians.opacities[0]=torch.squeeze(opacities_masked)

            newgaussians= Gaussians(
                means=means_masked_unshifted,
                covariances=covariance_masked,
                # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
                harmonics=harmonics_masked,
                opacities=opacities_masked,
                # Note: These aren't yet rotated into world space, but they're only used for
                # exporting Gaussians to ply files. This needs to be fixed...
                scales=scales_masked,
                rotations=rotations_masked,
            )
            self.render_splat(newgaussians,pred_all_extrinsic,pred_all_intrinsic)
            '''interpolated_output = self.model.decoder.forward(
                gaussians,
                pred_all_extrinsic,
                pred_all_intrinsic.float(),
                torch.ones(1, 2, device=pred_all_extrinsic.device) * 0.1,
                torch.ones(1, 2, device=pred_all_extrinsic.device) * 100,
                (h, w),
            )'''
        return

    def submap_gs_stitching_pipeline(self):
        config = Config

        for j in range(len(self.centers)):
            center = self.centers[j]
            path = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled/gaussiansmasked" + str(
                j) + ".ply"
            ply = PlyData.read(path)
            # render1=render_splatt(ply,pred_intrinsics[0],pred_extrinsics[0])
            # plt.imshow(render1, interpolation='nearest')
            # plt.show()
            # cam1_extrinsics = submap.inverted_poses[img_nr_1]
            # if j == 0:
            #    corrected_verts = transform_gaussians(ply,cam1_extrinsics[0:3,0:3],cam1_extrinsics[0:3,3:].reshape((3,)))
            # else:
            #    corrected_verts = np.concatenate([corrected_verts,transform_gaussians(ply,cam1_extrinsics[0:3,0:3],cam1_extrinsics[0:3,3:].reshape((3,)))])
            vetices = ply["vertex"].data
            #vetices["x"] = vetices["x"] + center[0]
            #vetices["y"] = vetices["y"] + center[1]
            #vetices["z"] = vetices["z"] + center[2]
            crop_paths=list(filter(lambda x: x.startswith("center"+str(j)+"image"), sorted(list(self.center_shiftdict.keys()))))
            crop_paths=[crop_paths[0],crop_paths[-1]]
            imagenr1 = int(crop_paths[0].removeprefix("center" + str(j) + "image"))
            extrinsics1 = self.inverted_poses[imagenr1]
            #extrinsics1 = self.poses[imagenr1]
            rotated_verts =self.transform_gaussians(vetices,extrinsics1[0:3,0:3])
            #rotated_verts =vetices
            rotated_verts["x"] = rotated_verts["x"] + center[0]
            rotated_verts["y"] = rotated_verts["y"] + center[1]
            rotated_verts["z"] = rotated_verts["z"] + center[2]
            if j == 0:
                corrected_verts = rotated_verts
            else:
                corrected_verts = np.concatenate([corrected_verts, rotated_verts])
            print(j)

        ply_out = PlyData([
            PlyElement.describe(corrected_verts, "vertex")
        ], text=False)
        ply_out.write(
            "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussiansubmapstitchseq0invex.ply")
        return

    def load_poses(self,path: Path) -> np.ndarray:
        """KITTI poses.txt (3×4 row-major per line) → (N, 4, 4) T_world_cam."""
        poses = []
        with open(path) as f:
            for line in f:
                vals = np.fromstring(line.strip(), sep=' ')
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = vals.reshape(3, 4)
                poses.append(T)
        return np.stack(poses)

    def invert_poses(self):
        inv =np.copy(self.poses)
        for i in range(len(self.poses)):
            inv[i]= np.linalg.inv(inv[i])

        return inv

    def fetch_images(self):
        i = self.frame_start
        imges = []
        while i <self.frame_end:
            if i < 10:
                imges.append(
                    self.config.image_path+"/00000" + str(i) + ".png")
            elif (i < 100) and (i > 9):
                imges.append(
                    self.config.image_path + "/0000" + str(i) + ".png")
            else:
                imges.append(
                    self.config.image_path + "/000" + str(i) + ".png")
            i+=1
        for i in range(len(imges)):
            img=Image.open(imges[i])
            if img.mode == 'RGBA':
                # Convert RGBA to RGB by removing alpha channel
                img = img.convert('RGB')
            imges[i] = np.asarray(img)

        return imges

    def fetch_centroids(self,radius):
        pcd_masked = np.copy(self.pcd)
        vol_origins = []
        for i in pcd_masked:
            flag = 0
            for j in vol_origins:
                if np.linalg.norm(np.array([i[0], i[1], i[2]]) - np.array([j[0], j[1], j[2]])) <= radius*1.5:
                    flag = 1
                    break
            if flag == 1:
                continue

            vol_origins.append(i)
        return vol_origins

    def project_centers(self, image_nr):
        """
        Project world-frame vertices into the camera defined by T_cw and global K.

        Returns
        -------
        projected_colors : (H, W, 3) uint8   mesh colour per pixel (black = no data)
        projection_ids   : (H, W)   int32    vertex .id per pixel  (−1   = no data)
        """
        N = len(self.centers)
        pts_c = (self.inverted_poses[image_nr] @ np.hstack([self.centers, np.ones((N, 1))]).T).T
        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

        u = (self.K[0, 0] * X / Z + self.K[0, 2]).astype(np.int32)
        v = (self.K[1, 1] * Y / Z + self.K[1, 2]).astype(np.int32)

        in_img = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        #u, v, Z = u[in_img], v[in_img], Z[in_img]


        return u, v, in_img

    def extract_window_no_padding(self, image, original_center_x, original_center_y, size=224,plot=True):
        """
        Extract a size x size window centered as close as possible to
        (center_x, center_y), shifting the window if near borders.
        """
        half = size // 2

        h, w = image.shape[:2]

        # Shift center so the window fits completely
        center_x = np.clip(original_center_x, half, w - half)
        center_y = np.clip(original_center_y, half, h - half)
        centershift_x, centershift_y = center_x - original_center_x, center_y - original_center_y


        x1 = center_x - half
        x2 = center_x + half
        y1 = center_y - half
        y2 = center_y + half
        crop = image[y1:y2, x1:x2]
        if plot:
            image1 = crop.copy()
            image2 = image.copy()
            for j in range(2):
                image1[114-centershift_y][114-centershift_x] = [255, 0, 0]
                image1[114-centershift_y + j][114-centershift_x] = [255, 0, 0]
                image1[114-centershift_y - j][114-centershift_x] = [255, 0, 0]
                image1[114-centershift_y][114-centershift_x + j] = [255, 0, 0]
                image1[114-centershift_y][114-centershift_x - j] = [255, 0, 0]
                image1[114-centershift_y + j][114-centershift_x+j] = [255, 0, 0]
                image1[114-centershift_y - j][114-centershift_x - j] = [255, 0, 0]
                image1[114-centershift_y- j][114-centershift_x + j] = [255, 0, 0]
                image1[114-centershift_y+ j][114-centershift_x - j] = [255, 0, 0]
                image2[original_center_y][original_center_x] = [255, 0, 0]
                image2[original_center_y + j][original_center_x] = [255, 0, 0]
                image2[original_center_y - j][original_center_x] = [255, 0, 0]
                image2[original_center_y][original_center_x + j] = [255, 0, 0]
                image2[original_center_y][original_center_x - j] = [255, 0, 0]
                image2[original_center_y + j][original_center_x+j] = [255, 0, 0]
                image2[original_center_y - j][original_center_x - j] = [255, 0, 0]
                image2[original_center_y- j][original_center_x + j] = [255, 0, 0]
                image2[original_center_y+ j][original_center_x - j] = [255, 0, 0]
            f, axarr = plt.subplots(2, 1)
            axarr[0].imshow(image1)
            axarr[1].imshow(image2)

            plt.show()

        return crop,(centershift_x,centershift_y)

    def transform_gaussians(self,ply,R,translation=0):
        #vertex = ply["vertex"].data
        vertex=ply
        rotation = Rotation.from_matrix(R)
        xyz = np.stack([
            vertex["x"],
            vertex["y"],
            vertex["z"]
        ], axis=1)

        # Quaternion fields used by Gaussian Splatting
        quat = np.stack([
            vertex["rot_0"],
            vertex["rot_1"],
            vertex["rot_2"],
            vertex["rot_3"]
        ], axis=1)

        # Transform positions

        #xyz_new = rotation.apply(xyz) + translation
        xyz_new = rotation.apply(xyz)

        # Transform orientations

        rot_scene = rotation

        quat_scipy = np.column_stack([
            quat[:, 1],
            quat[:, 2],
            quat[:, 3],
            quat[:, 0],
        ])

        gaussian_rot = Rotation.from_quat(quat_scipy)

        new_rot = rot_scene * gaussian_rot

        new_quat = new_rot.as_quat()

        quat_out = np.column_stack([
            new_quat[:, 3],  # w
            new_quat[:, 0],  # x
            new_quat[:, 1],  # y
            new_quat[:, 2],  # z
        ])

        # Copy vertex data
        new_vertex = vertex.copy()

        new_vertex["x"] = xyz_new[:, 0]
        new_vertex["y"] = xyz_new[:, 1]
        new_vertex["z"] = xyz_new[:, 2]

        new_vertex["rot_0"] = quat_out[:, 0]
        new_vertex["rot_1"] = quat_out[:, 1]
        new_vertex["rot_2"] = quat_out[:, 2]
        new_vertex["rot_3"] = quat_out[:, 3]

        return new_vertex





def main():
    meshpath = "/home/luis/Desktop/SABERGUIDE/KITTI/kitti_seq00_gt/mesh_kf000001.ply"
    framestart=0
    frameend=41
    submap = Submap(Config, framestart, frameend)
    submap.init_geometry(meshpath)
    submap.init_model()
    submap.init_model_inputs()
    submap.pipeline_foward()
    #submap.submap_gs_stitching_pipeline()
    #submap_pipeline(meshpath)
    #submap_gs_stitching_pipeline(meshpath)
    return


if __name__ == '__main__':
    sys.exit(main())