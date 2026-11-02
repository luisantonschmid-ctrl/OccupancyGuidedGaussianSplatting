from src.model.types import Gaussians
from src.model.encoder.common.gaussians import build_covariance
from math import sqrt

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
        self.intrinsics_crop = K.copy()
        self.intrinsics_crop[0,2]=112
        self.intrinsics_crop[1,2]=112
        self.R = r2
        self.T = t2
        self.img_h =config.img_h
        self.img_w = config.img_w
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.vol_radius = 2
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
        self.crop_starts_global = {}
        for j in range(len(self.centers)):
            for k in range(self.frame_end):
                if self.inside_mask[k][j]:
                    #print(self.im_u[k][j], self.im_v[k][j])
                    crop, centershift, start_points = self.extract_window_no_padding(self.images[k], int(self.im_u[k][j]),
                                                                         int(self.im_v[k][j]), plot=False)
                    self.center_shiftdict[str("center" + str(j) + "image" + str(k))] = {"x_shift": str(centershift[0]),
                                                                                        "y_shift": str(centershift[1])}
                    self.crop_starts_global[str("center" + str(j) + "image" + str(k))] = {"x": str(start_points[0]),
                                                                                   "y": str(start_points[1])}
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


    def render_splat_ret_img(self,means,scale,rot,harmonics, opacity,covariance,extrinsics,intrinsics,h=224*2,w=224*2):

        extrinsics_tensor = torch.tensor(np.linalg.inv(extrinsics), dtype=torch.float32, device="cuda")[None]
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device="cuda")[None]
        background_color = torch.tensor(np.array([1, 1, 1]), dtype=torch.float32, device="cuda")[None][0]
        feature = harmonics.permute(0, 1, 3, 2).contiguous()[0].float()
        rendering, alpha, _ = rasterization(means[0].float(), rot[0].float(),
                                            scale[0].float(), opacity[0].float(), feature,
                                            extrinsics_tensor.float(), intrinsics_tensor.float(), w, h,
                                            sh_degree=(int(sqrt(feature.shape[-2])) - 1),
                                            # near_plane=near[i].mean(), far_plane=far[i].mean(),
                                            render_mode="RGB+D", packed=False,
                                            near_plane=1e-10,
                                            backgrounds=background_color.unsqueeze(0).repeat(1, 1),
                                            radius_clip=0.1,
                                            covars=covariance[0].float(),
                                            rasterize_mode='classic')  # (V, H, W, 3)
        rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
        rendering_img = (rendering_img.clamp(0.0, 1.0).cpu().detach().numpy()[0] * 255).astype(np.uint8)
        return rendering_img

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


            crop_startx1=int(self.crop_starts_global[crop_paths[0]]['x'])
            crop_starty1=int(self.crop_starts_global[crop_paths[0]]['y'])
            crop_startx2=int(self.crop_starts_global[crop_paths[1]]['x'])
            crop_starty2=int(self.crop_starts_global[crop_paths[1]]['y'])

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

            #export_ply(gaussians.means[0], torch.squeeze(torch.tensor(gaussians.scales[0].cpu().detach().numpy(), dtype=torch.float32, device="cuda")[None]), gaussians.rotations[0], gaussians.harmonics[0],
            #           gaussians.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled') / ("gaussian" + str(i) + ".ply"))
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
            mask=np.linalg.norm(means - exactcenternp, axis=1)<= radius*2
            #mask = np.sum((means - np.array([exact_center])) ** 2, axis=1) <= radius**2
            #means = np.concatenate((means,np.array([xyz1])),axis=0)

            means_masked_unshifted=torch.tensor(means[mask], dtype=torch.float32, device="cuda")[None]

            means_premasked =means[mask]- np.array([exact_center])
            means_masked=torch.tensor(means_premasked/2, dtype=torch.float32, device="cuda")[None]
            scales =gaussians.scales[0]
            scales_masked = torch.tensor(scales.cpu().detach().numpy()[mask]*depths_scalar1/2, dtype=torch.float32, device="cuda")[None]
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
            #export_ply(torch.squeeze(means_masked), torch.squeeze(scales_masked), torch.squeeze(rotations_masked), torch.squeeze(harmonics_masked),
            #           torch.squeeze(opacities_masked), Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians2m2picsscaled') / ("gaussiansmasked"+str(i)+".ply"))
            pred_all_extrinsic = pred_context_pose['extrinsic']

            pred_all_intrinsic = pred_context_pose['intrinsic']


            '''            rendering, alpha, _ = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                                test_w2c_i[j:j + 1], test_intr_i[j:j + 1], W, H, sh_degree=sh_degree,
                                                # near_plane=near[i].mean(), far_plane=far[i].mean(),
                                                render_mode="RGB+D", packed=False,
                                                near_plane=1e-10,
                                                backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                                                radius_clip=0.1,
                                                covars=covar_i,
                                                rasterize_mode='classic')  # (V, H, W, 3)
            rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
            rendering_img = rendering_img.clamp(0.0, 1.0)'''

            gaussians_zero_centered= Gaussians(
                means=means_masked,
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


            #self.render_splat(newgaussians, pred_all_extrinsic, pred_all_intrinsic, image1, image2)
            predicted_extrinsics =pred_all_extrinsic.cpu().detach().numpy()[0]
            predicted_intrinsics = pred_all_intrinsic.cpu().detach().numpy()[0]
            xyzcords_up,xyzrgb_up = self.center_substitute(np.array(exact_center))
            v1_mesh_unposed = self.project_mesh(xyzcords_up, xyzrgb_up, np.linalg.inv(predicted_extrinsics[0]), predicted_intrinsics[0]*448)
            v2_mesh_unposed = self.project_mesh(xyzcords_up, xyzrgb_up, np.linalg.inv(predicted_extrinsics[1]), predicted_intrinsics[1]*448)


            v1_gs_unposed = self.render_splat_ret_img(means=means_masked_unshifted,
                                                       scale=scales_masked,
                                                       rot=rotations_masked,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=predicted_extrinsics[0],
                                                       intrinsics=predicted_intrinsics[0]*448,
                                                       h=448,
                                                       w=448)
            v2_gs_unposed = self.render_splat_ret_img(means=means_masked_unshifted,
                                                       scale=scales_masked,
                                                       rot=rotations_masked,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=predicted_extrinsics[1],
                                                       intrinsics=predicted_intrinsics[1]*448,
                                                       h=448,
                                                       w=448)

            v1_gs_posed = self.render_splat_ret_img(means=means_shifted_origin,
                                                       scale=scales_masked,
                                                       rot=rotations_masked,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics1,
                                                       intrinsics=self.intrinsics_crop,
                                                       h=448,
                                                       w=448)
            v2_gs_posed = self.render_splat_ret_img(means=means_shifted_origin,
                                                       scale=scales_masked,
                                                       rot=rotations_masked,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics2,
                                                       intrinsics=self.intrinsics_crop,
                                                       h=448,
                                                       w=448)
            v1_gs_posed_ogview = self.render_splat_ret_img(means=means_shifted_origin,
                                                       scale=scales_masked,
                                                       rot=rotations_masked,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics1,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            v2_gs_posed_ogview = self.render_splat_ret_img(means=means_shifted_origin,
                                                       scale=scales_masked,
                                                       rot=rotations_masked,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics2,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            xyzcords_p,xyzrgb_p = self.center_substitute(np.array(self.centers[i]))

            v1_mesh_posed = self.project_mesh(xyzcords_p, xyzrgb_p, extrinsics_inv1, self.K,IMG_W=self.img_w,IMG_H=self.img_h)
            v2_mesh_posed = self.project_mesh(xyzcords_p, xyzrgb_p, extrinsics_inv2, self.K,IMG_W=self.img_w,IMG_H=self.img_h)

            #initial_pose=np.array([0,0,0,self.centers[i][0],self.centers[i][1],self.centers[i][2]])
            initial_pose=np.array([0,0,0,distance1])

            sub_object_verts_soll, sub_object_rgb_soll = self.center_substitute(np.array(exact_center))

            u_soll, v_soll, in_img = self.project_points_twod(sub_object_verts_soll,np.linalg.inv(predicted_extrinsics[0]),predicted_intrinsics[0]*448)

            pix_soll =np.column_stack(((u_soll/2)+crop_startx1,(v_soll/2)+crop_starty1))
            #pix_soll =np.column_stack((u_soll,v_soll))

            origin,direction = self.pixel_to_world_ray(pix_soll[-1],self.K,extrinsics1)
            wp_center=origin+(direction*distance1)
            ini_t = np.array([origin[0],origin[1],origin[2],direction[0],direction[1],direction[2]])

            sub_object_verts_ist, sub_object_rgb_ist = self.center_substitute2(np.array([0,0,0]))

            T,result = self.optim_pose(reference_pix=pix_soll,
                                       substitute_mesh_verts=sub_object_verts_ist,
                                       initial_translate=ini_t,
                                      initial_rotate=initial_pose,
                                      extrinsics=extrinsics_inv1,
                                      intrinsics=self.K)
            '''            T,result = self.optim_pose(reference_pix=pix_soll,
                                       substitute_mesh_verts=sub_object_verts_ist,
                                       #initial_translate=,
                                      initial_pose=initial_pose,
                                      extrinsics=extrinsics_inv1,
                                      intrinsics=self.K)'''
            rotated = sub_object_verts_ist @ T[0:3,0:3].T
            translated = rotated + T[0:3,3]
            print('optimizedCenter: ',translated[-1])
            v1_mesh_optimized = self.project_mesh(translated, sub_object_rgb_ist, extrinsics_inv1, self.K,IMG_W=self.img_w,IMG_H=self.img_h)
            v2_mesh_optimized = self.project_mesh(translated, sub_object_rgb_ist, extrinsics_inv2, self.K,IMG_W=self.img_w,IMG_H=self.img_h)
            gaussians_optimized = self.transform_gaussians_any(gaussians_zero_centered,T[0:3,0:3],T[0:3,3])
            gaussians_extrinsics = self.transform_gaussians_any(gaussians_zero_centered,extrinsics1[0:3,0:3],self.centers[i])
            gaussians_invextrinsics =self.transform_gaussians_any(gaussians_zero_centered,extrinsics_inv1[0:3,0:3],self.centers[i])

            #gaussians_optimized = self.transform_gaussians_any(gaussians_zero_centered, T[0:3, 0:3], self.centers[i])
            v1_gs_optimized_ogview = self.render_splat_ret_img(means=gaussians_optimized.means,
                                                       scale=scales_masked,
                                                       rot=gaussians_optimized.rotations,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics1,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            v2_gs_optimized_ogview = self.render_splat_ret_img(means=gaussians_optimized.means,
                                                       scale=scales_masked,
                                                       rot=gaussians_optimized.rotations,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics2,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            v1_gs_optimized_ogview_extrinsics = self.render_splat_ret_img(means=gaussians_extrinsics.means,
                                                       scale=scales_masked,
                                                       rot=gaussians_extrinsics.rotations,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics1,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            v2_gs_optimized_ogview_extrinsics = self.render_splat_ret_img(means=gaussians_extrinsics.means,
                                                       scale=scales_masked,
                                                       rot=gaussians_extrinsics.rotations,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics2,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            v1_gs_optimized_ogview_invextrinsics = self.render_splat_ret_img(means=gaussians_invextrinsics.means,
                                                       scale=scales_masked,
                                                       rot=gaussians_invextrinsics.rotations,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics1,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            v2_gs_optimized_ogview_invextrinsics = self.render_splat_ret_img(means=gaussians_invextrinsics.means,
                                                       scale=scales_masked,
                                                       rot=gaussians_invextrinsics.rotations,
                                                       harmonics=harmonics_masked,
                                                       opacity=opacities_masked,
                                                       covariance=covariance_masked,
                                                       extrinsics=extrinsics2,
                                                       intrinsics=self.K,
                                                       h=self.img_h,
                                                       w=self.img_w)
            export_ply(gaussians_optimized.means[0], gaussians_optimized.scales[0], gaussians_optimized.rotations[0], gaussians_optimized.harmonics[0],
                       gaussians_optimized.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians'+str(self.vol_radius)+'m2picsoptim') / ("gaussiansoptim"+str(i)+".ply"))

            f, axarr = plt.subplots(2, 5)
            axarr[0, 0].imshow(self.images[imagenr1][crop_starty1:crop_starty1+224,crop_startx1:crop_startx1+224,:])
            axarr[0, 0].set_title('view1')
            axarr[0, 1].imshow(v1_gs_optimized_ogview_extrinsics[crop_starty1:crop_starty1+224,crop_startx1:crop_startx1+224,:])
            axarr[0, 1].set_title('v1_gs_optimized_ogview_extrinsics')
            axarr[0, 2].imshow(v1_gs_optimized_ogview[crop_starty1:crop_starty1+224,crop_startx1:crop_startx1+224,:])
            axarr[0, 2].set_title('v1_gs_optimized_ogview')
            axarr[0, 3].imshow(v1_gs_optimized_ogview_invextrinsics[crop_starty1:crop_starty1+224,crop_startx1:crop_startx1+224,:])
            axarr[0, 3].set_title('v1_gs_optimized_ogview_invextrinsics')
            axarr[1, 0].imshow(self.images[imagenr2][crop_starty2:crop_starty2+224,crop_startx2:crop_startx2+224,:])
            axarr[1, 0].set_title('view2')
            axarr[1, 1].imshow(v2_gs_optimized_ogview_extrinsics[crop_starty1:crop_starty1+224,crop_startx1:crop_startx1+224,:])
            axarr[1, 1].set_title('v2_gs_optimized_ogview_extrinsics')
            axarr[1, 2].imshow(v2_gs_optimized_ogview[crop_starty2:crop_starty2+224,crop_startx2:crop_startx2+224,:])
            axarr[1, 2].set_title('v2_gs_optimized_ogview')
            axarr[1, 3].imshow(v2_gs_optimized_ogview_invextrinsics[crop_starty1:crop_starty1+224,crop_startx1:crop_startx1+224,:])
            axarr[1, 3].set_title('v2_gs_optimized_ogview_invextrinsics')
            plt.show()
            time.sleep(1)

        return

    def pixel_to_world_ray(self,pixel, K, extrinsics):
        """
        Convert a pixel into a ray in world coordinates.

        Parameters
        ----------
        pixel : array-like
            [u, v] pixel coordinates

        K : (3, 3) ndarray
            Camera intrinsic matrix

        R : (3, 3) ndarray
            Camera-to-world rotation matrix

        t : (3,) ndarray
            Camera-to-world translation

        Returns
        -------
        origin : (3,) ndarray
            Camera position in world coordinates

        direction : (3,) ndarray
            Normalized ray direction in world coordinates
        """
        R, t = extrinsics[0:3, 0:3], extrinsics[0:3, 3]
        # Pixel -> camera ray
        u, v = pixel
        pixel_h = np.array([u, v, 1.0])

        direction_cam = np.linalg.inv(K) @ pixel_h
        direction_cam /= np.linalg.norm(direction_cam)

        # Camera ray -> world ray
        direction_world = R @ direction_cam
        direction_world /= np.linalg.norm(direction_world)

        # Camera position in world coordinates
        origin_world = t

        return origin_world, direction_world

    def transform_points(self,points, rotation,translation):
        rotated = points@rotation.T
        translated = rotated +translation
        return translated

    def optim_pose(self, reference_pix, initial_translate,initial_rotate, substitute_mesh_verts, extrinsics, intrinsics):
    #def optim_pose(self, reference_pix, initial_pose, substitute_mesh_verts, extrinsics, intrinsics):

        def render_object(transformation,substitute_mesh_verts,extrinsics,intrinsics):
            # Apply transformation to your object
            # Render it
            # Return image
            rotated = substitute_mesh_verts @ transformation[0:3,0:3].T
            translated = rotated + transformation[0:3,3]
            #IMG_W = 224 * 2
            #IMG_H = 224 * 2
            N = len(translated)
            pts_c = (extrinsics @ np.hstack([translated, np.ones((N, 1))]).T).T
            X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

            u = (intrinsics[0, 0] * X / Z + intrinsics[0, 2]).astype(np.int32)
            v = (intrinsics[1, 1] * Y / Z + intrinsics[1, 2]).astype(np.int32)

            #in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
            return np.column_stack((u, v))

        def image_loss(soll, ist):
            loss = 0
            for i in range(len(soll)):
                soll_pt=soll[i]
                ist_pt=ist[i]
                loss+= np.linalg.norm(soll_pt - ist_pt)

            return loss

        optimizer = PoseOptimizer(
            reference_image=reference_pix,
            render_fn=render_object,
            loss_fn=image_loss,
            substitute_mesh_verts = substitute_mesh_verts,
            extrinsics = extrinsics,
            intrinsics = intrinsics,
            initial_translate=initial_translate
        )
        T, result = optimizer.optimize(initial_rotate)
        #T, result = optimizer.optimize(initial_pose)

        print(T)
        print("Loss:", result.fun)

        return T,result

    def project_points_twod(self,points, extrinsics, intrinsics):

        N = len(points)
        pts_c = (extrinsics @ np.hstack([points, np.ones((N, 1))]).T).T
        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

        u = (intrinsics[0, 0] * X / Z + intrinsics[0, 2]).astype(np.int32)
        v = (intrinsics[1, 1] * Y / Z + intrinsics[1, 2]).astype(np.int32)

        in_img = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        #u, v, Z = u[in_img], v[in_img], Z[in_img]


        return u, v, in_img


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

    #def center_substitute(self,centervert):

        #radius=self.vol_radius

        #xyzcords=np.array([centervert + np.array([radius, 0, 0]),centervert + np.array([0, radius, 0]),centervert + np.array([0, 0, radius]),centervert + np.array([0, 0, 0])])
        #xyzrgb =np.array([np.array([254, 0, 0]),np.array([0, 254, 0]),np.array([0, 0, 254]),np.array([254, 254, 254])])
        #return xyzcords, xyzrgb

    def center_substitute(self,centervert):

        radius=self.vol_radius

        xyzcords=np.array([centervert + np.array([radius, 0, 0]),centervert + np.array([-radius, 0, 0]),centervert + np.array([0, radius, 0]),centervert + np.array([0, -radius, 0]),centervert + np.array([0, 0, radius]),centervert + np.array([0, 0, -radius]),centervert + np.array([radius*2, 0, 0]),centervert + np.array([-radius*2, 0, 0]),centervert + np.array([0, radius*2, 0]),centervert + np.array([0, -radius*2, 0]),centervert + np.array([0, 0, radius*2]),centervert + np.array([0, 0, -radius*2]),centervert + np.array([0, 0, 0])])
        xyzrgb =np.array([np.array([254, 0, 0]),np.array([254, 0, 0]),np.array([0, 254, 0]),np.array([0, 254, 0]),np.array([0, 0, 254]),np.array([0, 0, 254]),np.array([254, 0, 0]),np.array([254, 0, 0]),np.array([0, 254, 0]),np.array([0, 254, 0]),np.array([0, 0, 254]),np.array([0, 0, 254]),np.array([254, 254, 254])])
        return xyzcords, xyzrgb
    def center_substitute2(self,centervert):

        radius=self.vol_radius

        xyzcords=np.array([centervert + np.array([radius/2, 0, 0]),centervert + np.array([-radius/2, 0, 0]),centervert + np.array([0, radius/2, 0]),centervert + np.array([0, -radius/2, 0]),centervert + np.array([0, 0, radius/2]),centervert + np.array([0, 0, -radius/2]),centervert + np.array([radius, 0, 0]),centervert + np.array([-radius, 0, 0]),centervert + np.array([0, radius, 0]),centervert + np.array([0, -radius, 0]),centervert + np.array([0, 0, radius]),centervert + np.array([0, 0, -radius]),centervert + np.array([0, 0, 0])])
        xyzrgb =np.array([np.array([254, 0, 0]),np.array([254, 0, 0]),np.array([0, 254, 0]),np.array([0, 254, 0]),np.array([0, 0, 254]),np.array([0, 0, 254]),np.array([254, 0, 0]),np.array([254, 0, 0]),np.array([0, 254, 0]),np.array([0, 254, 0]),np.array([0, 0, 254]),np.array([0, 0, 254]),np.array([254, 254, 254])])
        return xyzcords, xyzrgb
    '''def center_substitute(self,centervert):

        radius=self.vol_radius
        x=np.array([radius, 0, 0])
        xc=np.array([254, 0, 0])
        y=np.array([0, radius, 0])
        yc=np.array([0, 254, 0])
        z=np.array([0, 0, radius])
        zc=np.array([0, 0, 254])
        xyzcords=np.array([centervert,centervert+x*2,centervert-x*2,centervert+y*2,centervert-y*2,centervert+z*2,centervert-z*2,centervert+(x+y)*2,centervert-(x+y)*2,centervert+(x+z)*2,centervert-(x+z)*2,centervert+(y+z)*2,centervert-(y+z)*2])
        xyzrgb =np.array([np.array([254, 254, 254]),xc,xc,yc,yc,zc,zc,xc+yc,xc+yc,xc+zc,xc+zc,yc+zc,yc+zc])
        return xyzcords, xyzrgb'''
    def submap_gs_stitching_pipeline(self):
        config = Config

        for j in range(len(self.centers)):
            center = self.centers[j]
            extracted_center =self.vol_origins_extracted[j]
            path = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians"+str(self.vol_radius)+"m2picsoptim/gaussiansoptim" + str(
                j) + ".ply"
            ply = PlyData.read(path)

            vetices = ply["vertex"].data
            #vetices["x"] = vetices["x"] + center[0]
            #vetices["y"] = vetices["y"] + center[1]
            #vetices["z"] = vetices["z"] + center[2]
            #crop_paths=list(filter(lambda x: x.startswith("center"+str(j)+"image"), sorted(list(self.center_shiftdict.keys()))))
            #crop_paths=[crop_paths[0],crop_paths[-1]]
            #imagenr1 = int(crop_paths[0].removeprefix("center" + str(j) + "image"))
            #extrinsics1 = self.inverted_poses[imagenr1]
            #extrinsics1 = self.poses[imagenr1]
            #rotated_verts =self.transform_gaussians(vetices,extrinsics1[0:3,0:3])
            rotated_verts =vetices
            #rotated_verts["x"] = rotated_verts["x"] + center[0]
            #rotated_verts["y"] = rotated_verts["y"] + center[1]
            #rotated_verts["z"] = rotated_verts["z"] + center[2]
            if j == 0:
                corrected_verts = rotated_verts
            else:
                corrected_verts = np.concatenate([corrected_verts, rotated_verts])
            print(j)

        ply_out = PlyData([
            PlyElement.describe(corrected_verts, "vertex")
        ], text=False)
        ply_out.write(
            "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussiansubmapstitch"+str(self.vol_radius)+"moptimv5.ply")
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

        return crop,(centershift_x,centershift_y),(x1,y1)

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

    def transform_gaussians_any(self,gaussians,R,translation):
        #vertex = ply["vertex"].data
        #vertex=ply

        xyz = gaussians.means[0].cpu().detach().numpy()
        quat = gaussians.rotations[0].cpu().detach().numpy()

        rotation = Rotation.from_matrix(R)

        # Transform positions

        xyz_new = rotation.apply(xyz) + translation

        # Transform orientations

        gaussian_rot = Rotation.from_quat(quat)

        new_rot = rotation * gaussian_rot

        new_quat = new_rot.as_quat()

        means_tensor = torch.tensor(xyz_new, dtype=torch.float32, device="cuda")[None]
        rotations_tensor = torch.tensor(new_quat, dtype=torch.float32, device="cuda")[None]

        newgaussians = Gaussians(
            means=means_tensor,
            covariances=gaussians.covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=gaussians.harmonics,
            opacities=gaussians.opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=gaussians.scales,
            rotations=rotations_tensor
        )
        return newgaussians




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