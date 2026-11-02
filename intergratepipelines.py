from src.model.types import Gaussians
from src.model.encoder.common.gaussians import build_covariance

from pathlib import Path
import torch
import os
import sys
import pandas as pd
from AnySplat.src.misc.image_io import save_interpolated_video
from AnySplat.src.model.model.anysplat import AnySplat
from AnySplat.src.model.encoder.vggt.utils import rotation
#from src.utils.image import process_image
from src.model.ply_export import export_ply
import glob
from os import listdir
from os.path import isfile, join
import cv2
from optimizer_mesh import PoseOptimizer
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
from kittiseq10_config import Config
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
FAR_CLIP = 160.0  # metres — discard points farther than this
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
        self.vol_radius = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vol_origins_extracted = []


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
    def render_splat(self,gaussians,pred_all_extrinsic,pred_all_intrinsic,img1,img2,h=224*2,w=224*2):

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

        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(img1)
        axarr[0, 1].imshow(video[0,:,:,:])
        axarr[1, 0].imshow(img2)
        axarr[1, 1].imshow(video[1,:,:,:])
        plt.show()
        time.sleep(1)

        #    cv2.imwrite(crop_paths[googoldcounter].replace('cropped2m','cropped2mreconstructed'),frame)
        #    googoldcounter+=1
        return

    def render_splat_ret_img(self,gaussians,pred_all_extrinsic,pred_all_intrinsic,h=224*2,w=224*2):

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

        return video[0,:,:,:],video[1,:,:,:]

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
            extrinsics_inv1 = self.inverted_poses[imagenr1]
            extrinsics_inv2 = self.inverted_poses[imagenr2]

            self.model.encoder.extrinsics1 = extrinsics1
            self.model.encoder.extrinsics2 = extrinsics2
            intrinsicsfxy = 707.09
            self.model.encoder.intrinsicsfxy = intrinsicsfxy
            images = (images + 1) * 0.5
            gaussians, pred_context_pose = self.model.inference(images)

            depths_scalar1=self.model.encoder.depths_scalar1
            depths_scalar2=self.model.encoder.depths_scalar2
            mean_coord1 = self.model.encoder.meancoord1
            mean_coord2 = self.model.encoder.meancoord2
            depth_pixel1 = self.model.encoder.depths_of_pixel1
            depth_pixel2 = self.model.encoder.depths_of_pixel2

            export_ply(gaussians.means[0], torch.squeeze(torch.tensor(gaussians.scales[0].cpu().detach().numpy(), dtype=torch.float32, device="cuda")[None]), gaussians.rotations[0], gaussians.harmonics[0],
                       gaussians.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled') / ("gaussian" + str(i) + ".ply"))
            means_unscaled = gaussians.means[0].cpu().detach().numpy()
            #single_gaussian = self.create_single_gaussians(gaussians, means_unscaled, mean_coord1)

            means = means_unscaled*depths_scalar1
            radius=self.vol_radius

            #center = get_pixel_3d_coord(target_u1*2, target_v1*2, gaussians, 448, 448, view_idx=0)
            exact_center=[mean_coord1[0]*depths_scalar1, mean_coord1[1]*depths_scalar1, mean_coord1[2]*depths_scalar1]

            exactcenternp=np.array([exact_center])
            self.vol_origins_extracted.append(exact_center)

            means_shifted = means -exactcenternp
            print('ppredictedgscenter:',exactcenternp)
            #print(center)
            print('globalcenter:',self.centers[i])
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
            means_shifted_origin = torch.tensor(means_masked[0].cpu().detach().numpy()+np.array(self.centers[i]), dtype=torch.float32, device="cuda")[None]
            gaussians_shifted= Gaussians(
                means=means_shifted_origin,
                covariances=covariance_masked,
                # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
                harmonics=harmonics_masked,
                opacities=opacities_masked,
                # Note: These aren't yet rotated into world space, but they're only used for
                # exporting Gaussians to ply files. This needs to be fixed...
                scales=scales_masked,
                rotations=rotations_masked,
            )
            #self.render_splat(single_gaussian, pred_all_extrinsic, pred_all_intrinsic, image1, image2)

            #self.render_splat(newgaussians, pred_all_extrinsic, pred_all_intrinsic, image1, image2)
            v1_gs_unposed,v2_gs_unposed = self.render_splat_ret_img(newgaussians,pred_all_extrinsic,pred_all_intrinsic)
            predicted_extrinsics =pred_all_extrinsic.cpu().detach().numpy()[0]
            predicted_intrinsics = pred_all_intrinsic.cpu().detach().numpy()[0]
            xyzcords_up,xyzrgb_up = self.center_substitute(np.array(exact_center))
            v1_mesh_unposed = self.project_mesh(xyzcords_up, xyzrgb_up, np.linalg.inv(predicted_extrinsics[0]), predicted_intrinsics[0]*448)
            v2_mesh_unposed = self.project_mesh(xyzcords_up, xyzrgb_up, np.linalg.inv(predicted_extrinsics[1]), predicted_intrinsics[1]*448)

            intr=pred_all_intrinsic.cpu().detach().numpy()
            intr[0,0,0,0]=intrinsicsfxy/448
            intr[0,0,1,1]=intrinsicsfxy/448
            intr[0,1,0,0]=intrinsicsfxy/448
            intr[0,1,1,1]=intrinsicsfxy/448
            intrinsics_corrected = torch.tensor(intr[0, :, :,:], dtype=torch.float32, device="cuda")[None]

            v1_gs_posed,v2_gs_posed = self.render_splat_ret_img(gaussians_shifted,torch.tensor(np.array([extrinsics1,extrinsics2]), dtype=torch.float32, device="cuda")[None],intrinsics_corrected)

            xyzcords_p,xyzrgb_p = self.center_substitute(np.array(self.centers[i]))

            v1_mesh_posed = self.project_mesh(xyzcords_p, xyzrgb_p, extrinsics_inv1, intr[0][0]*448)
            v2_mesh_posed = self.project_mesh(xyzcords_p, xyzrgb_p, extrinsics_inv2, intr[0][1]*448)
            f, axarr = plt.subplots(2, 5)
            #f.update_annotations(font_size=12)
            #f.up
            #f.
            axarr[0, 0].imshow(image1)
            axarr[0, 0].set_title('view1')
            axarr[0, 1].imshow(v1_gs_unposed)
            axarr[0, 1].set_title('view1GsUnposed')
            axarr[0, 2].imshow(v1_mesh_unposed)
            axarr[0, 2].set_title('view1MeshUnposed')
            axarr[0, 3].imshow(v1_gs_posed)
            axarr[0, 3].set_title('view1GsPosed')
            axarr[0, 4].imshow(v1_mesh_posed)
            axarr[0, 4].set_title('view1MeshPosed')
            axarr[1, 0].imshow(image2)
            axarr[1, 0].set_title('view2')
            axarr[1, 1].imshow(v2_gs_unposed)
            axarr[1, 1].set_title('view2GsUnposed')
            axarr[1, 2].imshow(v2_mesh_unposed)
            axarr[1, 2].set_title('view2MeshUnposed')
            axarr[1, 3].imshow(v2_gs_posed)
            axarr[1, 3].set_title('view2GsPosed')
            axarr[1, 4].imshow(v2_mesh_posed)
            axarr[1, 4].set_title('view2MeshPosed')

            plt.show()
            time.sleep(1)
            initial_pose=np.array([0,0,0,self.centers[i][0],self.centers[i][1],self.centers[i][2]])
            sub_object_verts, sub_object_rgb = self.center_substitute(np.array([0,0,0]))

            T,result = self.optim_pose(reference_image=v1_mesh_unposed,
                                  initial_pose=initial_pose,
                                  substitute_mesh_verts=sub_object_verts,
                                  substitute_mesh_rgb=sub_object_rgb,
                                  extrinsics=extrinsics_inv1,
                                  intrinsics=intr[0][0]*448,
                                  near=NEAR_CLIP,
                                  far=FAR_CLIP,
                                  pt_size=POINT_SIZE)
            f, axarr = plt.subplots(2, 3)
            axarr[0, 0].imshow(v1_mesh_unposed)
            axarr[0, 1].imshow(v1_mesh_posed)
            xyzcords_optim =self.transform_points(sub_object_verts,T[0:3,0:3],T[0:3,3])
            v1_mesh_otimized = self.project_mesh(xyzcords_optim, sub_object_rgb, extrinsics_inv1, intr[0][0]*448)
            axarr[0, 2].imshow(v1_mesh_otimized)
            plt.show()
            time.sleep(1)

            #xyzcords_p_r = self.transform_points(sub_object,extrinsics1[0:3,0:3],self.centers[i])
            #xyzcords_p_r_inv = self.transform_points(sub_object,np.linalg.inv(extrinsics1[0:3,0:3]),self.centers[i])
            #xyzcords_p_r_free = self.transform_points(sub_object,np.array([[1,0,0],[0,0,np.pi*-1*0.25],[0,np.pi*0.25,0]]),self.centers[i])
            #v1_mesh_posed_r = self.project_mesh(xyzcords_p_r, subrgb, extrinsics_inv1, intr[0][0]*448)
            #v1_mesh_posed_r_inv = self.project_mesh(xyzcords_p_r_inv, subrgb, extrinsics_inv1, intr[0][0]*448)
            #custom_intrinsics=intr[0][0]*448
            #custom_intrinsics[0,2]=1000
            #custom_intrinsics[1,2]=1000
            #v1_mesh_posed_r_free = self.project_mesh(xyzcords_p_r_free, subrgb, extrinsics_inv1, custom_intrinsics,IMG_H=2000,IMG_W=2000)
            #f, axarr = plt.subplots(2, 3)
            #axarr[0, 0].imshow(v1_mesh_posed)
            #axarr[0, 1].imshow(v1_mesh_posed_r)
            #axarr[0, 2].imshow(v1_mesh_posed_r_inv)
            #axarr[1, 0].imshow(v1_mesh_posed_r_free)
            #plt.show()
            #time.sleep(1)

        return

    def transform_points(self,points, rotation,translation):
        rotated = points@rotation.T
        translated = rotated +translation
        return translated

    def optim_pose(self, reference_image,initial_pose,substitute_mesh_verts, substitute_mesh_rgb, extrinsics, intrinsics,near, far, pt_size):

        def render_object(transformation,substitute_mesh_verts, substitute_mesh_rgb,extrinsics,intrinsics,near,far,pt_size):
            # Apply transformation to your object
            # Render it
            # Return image
            rotated = substitute_mesh_verts @ transformation[0:3,0:3].T
            translated = rotated + transformation[0:3,3]
            T_cw = extrinsics
            K = intrinsics
            IMG_W = 224 * 2
            IMG_H = 224 * 2
            # IMG_W=224
            # IMG_H=224
            N = len(translated)
            pts_c = (T_cw @ np.hstack([translated, np.ones((N, 1))]).T).T  # (N, 4)
            X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

            valid = (Z > near) & (Z < far)
            X, Y, Z = X[valid], Y[valid], Z[valid]
            c = substitute_mesh_rgb[valid]

            u = (K[0, 0] * X / Z + K[0, 2]).astype(np.int32)
            v = (K[1, 1] * Y / Z + K[1, 2]).astype(np.int32)

            in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
            u, v, Z, c = u[in_img], v[in_img], Z[in_img], c[in_img]

            # Z-buffer: paint far-to-near so closer points overwrite
            order = np.argsort(Z)[::-1]
            u, v, c = u[order], v[order], c[order]

            proj_colors = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

            if pt_size <= 1:
                proj_colors[v, u] = c
            else:
                # Vectorised blob: loop over (2r+1)² offsets; each pass is fully vectorised
                r = pt_size
                dv_grid, du_grid = np.mgrid[-r:r + 1, -r:r + 1]
                for dv, du in zip(dv_grid.ravel(), du_grid.ravel()):
                    vv = np.clip(v + dv, 0, IMG_H - 1)
                    uu = np.clip(u + du, 0, IMG_W - 1)
                    proj_colors[vv, uu] = c

            return proj_colors

        def image_loss(reference, rendered):
            loss = 1
            empty=np.zeros_like(reference)
            ref_score = np.count_nonzero(reference)
            ren_score = np.count_nonzero(rendered)
            r = np.array([254, 0, 0])
            g = np.array([0, 254, 0])
            b = np.array([0, 0, 254])
            black = np.array([0, 0, 0])

            if ren_score==0:
                return 1

            if ref_score==ren_score:
                loss+=-0.2
            elif ref_score>ren_score:
                loss+=-ren_score/ref_score/5
            else:
                loss += -ref_score/ren_score/5
            n_ref_pipx=0
            x_ref=0
            y_ref=0
            n_ren_pipx=0
            x_ren=0
            y_ren=0

            for i in range(len(reference)):
                for j in range(len(reference[0])):
                    point_rendered =rendered[i][j]
                    point_reference = reference[i][j]
                    if not np.array_equal(point_rendered, black):
                        n_ren_pipx+=1
                        y_ren+=j
                        x_ren+=i
                    if not np.array_equal(point_reference, black):
                        n_ref_pipx+=1
                        y_ref+=j
                        x_ref+=i
            ren_moment = np.array([x_ren/n_ren_pipx,y_ren/n_ren_pipx])
            ref_moment = np.array([x_ref/n_ref_pipx,y_ref/n_ref_pipx])
            max_dist = np.max(np.array([np.linalg.norm(np.array([0,0]) - ref_moment),np.linalg.norm(np.array([448,0]) - ref_moment),np.linalg.norm(np.array([0,448]) - ref_moment),np.linalg.norm(np.array([448,448]) - ref_moment)]))
            dist = np.linalg.norm(ren_moment - ref_moment)
            loss += -dist/max_dist/5
            print(loss)
            return loss

        optimizer = PoseOptimizer(
            reference_image=reference_image,
            render_fn=render_object,
            loss_fn=image_loss,
            substitute_mesh_verts = substitute_mesh_verts,
            substitute_mesh_rgb = substitute_mesh_rgb,
            extrinsics = extrinsics,
            intrinsics = intrinsics,
            near = near,
            far = far,
            pt_size = pt_size
        )
        T, result = optimizer.optimize(initial_pose)

        print(T)
        print("Loss:", result.fun)

        return T,result

    def project_mesh(self, vertices, vert_colors, extrinsics, intrinsics,
                     near=NEAR_CLIP, far=FAR_CLIP, pt_size=POINT_SIZE,IMG_W=224*2,IMG_H=224*2):


        """
        Project world-frame vertices into the camera defined by T_cw and global K.

        Returns
        -------
        projected_colors : (H, W, 3) uint8   mesh colour per pixel (black = no data)
        projection_ids   : (H, W)   int32    vertex .id per pixel  (−1   = no data)
        """
        T_cw=extrinsics
        K =intrinsics
        #IMG_W=224
        #IMG_H=224
        N = len(vertices)
        pts_c = (T_cw @ np.hstack([vertices, np.ones((N, 1))]).T).T  # (N, 4)
        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

        valid = (Z > near) & (Z < far)
        X, Y, Z = X[valid], Y[valid], Z[valid]
        c = vert_colors[valid]

        u = (K[0, 0] * X / Z + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * Y / Z + K[1, 2]).astype(np.int32)

        in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        u, v, Z, c = u[in_img], v[in_img], Z[in_img], c[in_img]

        # Z-buffer: paint far-to-near so closer points overwrite
        order = np.argsort(Z)[::-1]
        u, v, c = u[order], v[order], c[order]

        proj_colors = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        if pt_size <= 1:
            proj_colors[v, u] = c
        else:
            # Vectorised blob: loop over (2r+1)² offsets; each pass is fully vectorised
            r = pt_size
            dv_grid, du_grid = np.mgrid[-r:r + 1, -r:r + 1]
            for dv, du in zip(dv_grid.ravel(), du_grid.ravel()):
                vv = np.clip(v + dv, 0, IMG_H - 1)
                uu = np.clip(u + du, 0, IMG_W - 1)
                proj_colors[vv, uu] = c

        return proj_colors

    def center_substitute(self,centervert):

        #centervert = np.array([2, 2, 10])

        xcoords = np.zeros((40, 3))
        xrgb = np.zeros((40, 3))
        ycoords = np.zeros((40, 3))
        yrgb = np.zeros((40, 3))
        zcoords = np.zeros((40, 3))
        zrgb = np.zeros((40, 3))

        for i in range(40):
            xcoords[i] = centervert + np.array([i * 0.1, 0, 0])
            xrgb[i][0] += 254
            ycoords[i] = centervert + np.array([0, i * 0.1, 0])
            yrgb[i][1] += 254
            zcoords[i] = centervert + np.array([0, 0, i * 0.1])
            zrgb[i][2] += 254

        xyzcords = np.vstack((xcoords, ycoords, zcoords))
        xyzrgb = np.vstack((xrgb, yrgb, zrgb))

        return xyzcords, xyzrgb

    def create_single_gaussians(self, gaussians, means_unscaled,mean_coord):
        dists = np.linalg.norm(means_unscaled - mean_coord, axis=1)
        mindist_index = np.argmin(dists)
        mindist_val = dists[mindist_index]
        unitshiftx=np.reshape(np.array([1,0,0]),(3,))
        unitshifty=np.reshape(np.array([0,1,0]),(3,))
        unitshiftz=np.reshape(np.array([0,0,1]),(3,))
        center_mean1 = gaussians.means[0][mindist_index].clone().cpu().detach().numpy()
        center_mean2 = np.array([center_mean1,center_mean1+unitshiftx,center_mean1+unitshifty,center_mean1+unitshiftz])
        center_mean3 = torch.tensor(center_mean2, dtype=torch.float32, device="cuda")[None]

        center_scales1 = gaussians.scales[0][mindist_index].clone().cpu().detach().numpy()*200
        center_scales2 = np.array([center_scales1,center_scales1,center_scales1,center_scales1])
        center_scales3 = torch.tensor(center_scales2, dtype=torch.float32, device="cuda")[None]

        center_covariances1 = gaussians.covariances[0][mindist_index].clone().cpu().detach().numpy()
        center_covariances2 = np.array([center_covariances1,center_covariances1,center_covariances1,center_covariances1])
        center_covariances3 = torch.tensor(center_covariances2, dtype=torch.float32, device="cuda")[None]

        rx=np.array([[1,0,0],[0,0,np.pi*-1*0.25],[0,np.pi*0.25,0]])
        ry=np.array([[0,0,np.pi*0.25],[0,1,0],[np.pi*-1*0.25,0,0]])
        rz=np.array([[0,np.pi*-1*0.25,0],[np.pi*0.25,0,0],[0,0,1]])
        rzero=np.array([[1,0,0],[0,1,0],[0,0,1]])
        rapply= rz@rzero

        center_rotations1 = rotation.quat_to_mat(gaussians.rotations[0][mindist_index].clone()).cpu().detach().numpy()
        center_rotations2 = np.array([center_rotations1 ,rx @ center_rotations1, ry @ center_rotations1, rz @ center_rotations1])
        center_rotations3 = rotation.mat_to_quat(torch.tensor(center_rotations2, dtype=torch.float32, device="cuda")[None])



        center_opacities1 = gaussians.opacities[0][mindist_index].clone().cpu().detach().numpy()
        center_opacities2 = np.array([1,1,1,1])
        center_opacities3 = torch.tensor(center_opacities2, dtype=torch.float32, device="cuda")[None]

        center_harmonics1 = gaussians.harmonics[0][mindist_index].clone().cpu().detach().numpy()
        center_harmonics2 = np.array([center_harmonics1,center_harmonics1,center_harmonics1,center_harmonics1])
        center_harmonics3 = torch.tensor(center_harmonics2, dtype=torch.float32, device="cuda")[None]

        single_gaussian = Gaussians(
                means=center_mean3,
                covariances=center_covariances3,
                harmonics=center_harmonics3,
                opacities=center_opacities3,
                scales=center_scales3,
                rotations=center_rotations3,
            )
        export_ply(torch.squeeze(center_mean3), torch.squeeze(center_scales3), torch.squeeze(center_rotations3),
                   torch.squeeze(center_harmonics3),
                   torch.squeeze(center_opacities3), Path(
                '/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output') / (
                               "gaussiansbs" + str(66) + ".ply"))
        return single_gaussian

    def submap_gs_stitching_pipeline(self):
        config = Config

        for j in range(len(self.centers)):
            center = self.centers[j]
            extracted_center =self.vol_origins_extracted[j]
            path = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled/gaussiansmasked" + str(
                j) + ".ply"
            ply = PlyData.read(path)

            vetices = ply["vertex"].data
            #vetices["x"] = vetices["x"] + center[0]
            #vetices["y"] = vetices["y"] + center[1]
            #vetices["z"] = vetices["z"] + center[2]
            crop_paths=list(filter(lambda x: x.startswith("center"+str(j)+"image"), sorted(list(self.center_shiftdict.keys()))))
            crop_paths=[crop_paths[0],crop_paths[-1]]
            imagenr1 = int(crop_paths[0].removeprefix("center" + str(j) + "image"))
            extrinsics1 = self.inverted_poses[imagenr1]
            #extrinsics1 = self.poses[imagenr1]
            #rotated_verts =self.transform_gaussians(vetices,extrinsics1[0:3,0:3])
            rotated_verts =vetices
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
            "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussiansubmapstitch4m.ply")
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
                if np.linalg.norm(np.array([i[0], i[1], i[2]]) - np.array([j[0], j[1], j[2]])) <= radius:
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
    meshpath = "/home/luis/Desktop/SABERGUIDE/KITTI/kitti_seq10_gt/mesh_kf000001.ply"
    framestart=0
    frameend=34
    submap = Submap(Config, framestart, frameend)
    submap.init_geometry(meshpath)
    submap.init_model()
    submap.init_model_inputs()
    submap.pipeline_foward()
    submap.submap_gs_stitching_pipeline()
    #submap_pipeline(meshpath)
    #submap_gs_stitching_pipeline(meshpath)
    return


if __name__ == '__main__':
    sys.exit(main())