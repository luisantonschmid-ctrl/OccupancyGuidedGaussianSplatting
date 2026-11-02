from src.model.types import Gaussians
from src.model.encoder.common.gaussians import build_covariance
from math import sqrt
from scipy.spatial.transform import Rotation as R
from AnySplat.src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from optimizer_mesh import PoseOptimizer
import sys
from plyfile import PlyData, PlyElement
import trimesh
import numpy as np
from kittiseq10_config import Config
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

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

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
        self.model.encoder.cfg.voxelize=False
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
        self.centers, self.volumes = self.discretize_pointcloud()
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
        self.subcenter_dict = {}
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
        iterator = np.arange(len(self.centers))
        iterator_subc = np.arange(len(self.centers))

        while iterator.size != 0:
            centernumber=iterator[0]
            iterator = iterator[1:]

            crop_path = list(filter(lambda x: x.startswith("center" + str(centernumber) + "image"), sorted(list(self.center_shiftdict.keys()))))[0]
            imagenr = int(crop_path.removeprefix("center" + str(centernumber) + "image"))
            u,v=self.im_u[imagenr][centernumber],self.im_v[imagenr][centernumber]
            centervert = self.centers[centernumber]
            crop = self.crop_dict[crop_path]['crop']
            crop_start = self.crop_starts_global[crop_path]
            self.subcenter_dict[centernumber] = {'pixel':[u,v], 'vert':centervert,'crop':crop}
            index = np.where(iterator_subc == centernumber)[0][0]
            iterator_subc=np.delete(iterator_subc, index)
            if iterator.size != 0:
                for i in iterator_subc:
                    if self.inside_mask[imagenr][i]:
                        sub_u, sub_v = int(self.im_u[imagenr][i]),int(self.im_v[imagenr][i])
                        if (int(crop_start['x'])<sub_u<int(crop_start['x'])+224)and(int(crop_start['y'])<sub_v<int(crop_start['y'])+224):
                            self.subcenter_dict[centernumber]['subcenter_'+str(i)]={'pixel':[sub_u,sub_v], 'vert':self.centers[i],'centernumber':i}
                            try:
                                index=np.where(iterator == i)[0][0]
                                #print(index)
                                iterator = np.delete(iterator, index)
                            except:
                                continue



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
        return rendering_img, rendering_depth.cpu().detach().numpy()[0]

    def pipeline_foward(self,visual_toggle=False):
        for key in self.subcenter_dict.keys():
        #for i in range(len(self.centers)):
            i = key
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
            print(self.im_u[imagenr1][i],self.im_v[imagenr1][i])
            self.model.encoder.pixeldepth1 = (target_u1,target_v1,distance1)
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

            original_pixel_coordinate = np.array([crop_startx1+target_u1,crop_starty1+target_v1])

            gaussians, pred_context_pose = self.model.inference(images)
            gaussians =Gaussians(
                means=torch.tensor(gaussians.means[0].cpu().detach().numpy()[0:448*448], dtype=torch.float32, device="cuda")[None],
                covariances=torch.tensor(gaussians.covariances[0].cpu().detach().numpy()[0:448*448], dtype=torch.float32, device="cuda")[None],
                harmonics=torch.tensor(gaussians.harmonics[0].cpu().detach().numpy()[0:448*448], dtype=torch.float32, device="cuda")[None],
                opacities=torch.tensor(gaussians.opacities[0].cpu().detach().numpy()[0:448*448], dtype=torch.float32, device="cuda")[None],
                scales=torch.tensor(gaussians.scales[0].cpu().detach().numpy()[0:448*448], dtype=torch.float32, device="cuda")[None],
                rotations=torch.tensor(gaussians.rotations[0].cpu().detach().numpy()[0:448*448], dtype=torch.float32, device="cuda")[None],
            )
            pred_all_extrinsic = pred_context_pose['extrinsic']

            pred_all_intrinsic = pred_context_pose['intrinsic']
            predicted_extrinsics =pred_all_extrinsic.cpu().detach().numpy()[0]
            predicted_intrinsics = pred_all_intrinsic.cpu().detach().numpy()[0]
            predicted_intrinsics_448 = predicted_intrinsics.copy()
            predicted_intrinsics_448[:,0:2,:]=predicted_intrinsics_448[:,0:2,:]*448

            inv_pred_ex1=np.linalg.inv(predicted_extrinsics[0])
            inv_pred_ex2=np.linalg.inv(predicted_extrinsics[1])
            cutom_intr = predicted_intrinsics_448.copy()
            cutom_intr[0,0,2]=112
            cutom_intr[0,1,2]=112
            cutom_intr[1,0,2]=112
            cutom_intr[1,1,2]=112

            depths_scalar1=self.model.encoder.depths_scalar1
            depths_scalar2=self.model.encoder.depths_scalar2
            mean_coord1 = self.model.encoder.meancoord1
            mean_coord2 = self.model.encoder.meancoord2
            depth_pixel1 = self.model.encoder.depths_of_pixel1
            depth_pixel2 = self.model.encoder.depths_of_pixel2

            means = gaussians.means[0].clone().cpu().detach().numpy()
            scales =gaussians.scales[0].clone().cpu().detach().numpy()
            rotations = gaussians.rotations[0].clone().cpu().detach().numpy()
            harmonics=gaussians.harmonics[0].clone().cpu().detach().numpy()
            opacities=gaussians.opacities[0].clone().cpu().detach().numpy()

            e1means=means@inv_pred_ex1[0:3,0:3].T+np.array([inv_pred_ex1[0:3,3]])
            e1means[:,2]=e1means[:,2]*2
            means_zscale=e1means@predicted_extrinsics[0][0:3,0:3].T+np.array([predicted_extrinsics[0][0:3,3]])

            means_zscale_t=torch.tensor(means_zscale, dtype=torch.float32, device="cuda")[None]
            scales_zscale_t = torch.tensor(scales, dtype=torch.float32, device="cuda")[None]
            rotations_zscale_t = torch.tensor(rotations, dtype=torch.float32, device="cuda")[None]
            harmonics_zscale_t = torch.tensor(harmonics, dtype=torch.float32, device="cuda")[None]
            opacities_zscale_t = torch.tensor(opacities, dtype=torch.float32, device="cuda")[None]
            covariance_zscale_t = build_covariance(scales_zscale_t,rotations_zscale_t)

            gaussians_z= Gaussians(
                means=means_zscale_t,
                covariances=covariance_zscale_t,
                harmonics=harmonics_zscale_t,
                opacities=opacities_zscale_t,
                scales=scales_zscale_t,
                rotations=rotations_zscale_t,
            )

            correct_gaussians, correct_gaussians_center = self.transform_gaussians_individual(gaussians_z,predicted_extrinsics[0],(i,imagenr1),(crop_startx1,crop_starty1), distance1,(target_u1,target_v1))

            correct_means = correct_gaussians.means[0].clone().cpu().detach().numpy()
            correct_scales =correct_gaussians.scales[0].clone().cpu().detach().numpy()
            correct_rotations = correct_gaussians.rotations[0].clone().cpu().detach().numpy()
            correct_harmonics=correct_gaussians.harmonics[0].clone().cpu().detach().numpy()
            correct_opacities=correct_gaussians.opacities[0].clone().cpu().detach().numpy()


            correct_means_t=torch.tensor(correct_means, dtype=torch.float32, device="cuda")[None]
            correct_scales_t = torch.tensor(correct_scales, dtype=torch.float32, device="cuda")[None]
            correct_rotations_t = torch.tensor(correct_rotations, dtype=torch.float32, device="cuda")[None]
            correct_harmonics_t = torch.tensor(correct_harmonics, dtype=torch.float32, device="cuda")[None]
            correct_opacities_t = torch.tensor(correct_opacities, dtype=torch.float32, device="cuda")[None]
            correct_covariances_t = build_covariance(correct_scales_t,correct_rotations_t)

            correct_mask =np.linalg.norm(correct_means - np.array(correct_gaussians_center), axis=1) <= self.vol_radius

            correct_means_tm=torch.tensor(correct_means[correct_mask], dtype=torch.float32, device="cuda")[None]
            correct_scales_tm = torch.tensor(correct_scales[correct_mask], dtype=torch.float32, device="cuda")[None]
            correct_rotations_tm = torch.tensor(correct_rotations[correct_mask], dtype=torch.float32, device="cuda")[None]
            correct_harmonics_tm = torch.tensor(correct_harmonics[correct_mask], dtype=torch.float32, device="cuda")[None]
            correct_opacities_tm = torch.tensor(correct_opacities[correct_mask], dtype=torch.float32, device="cuda")[None]
            correct_covariances_tm = build_covariance(correct_scales_tm,correct_rotations_tm)


            corrected_gaussians= Gaussians(
                means=correct_means_t,
                covariances=correct_covariances_t,
                harmonics=correct_harmonics_t,
                opacities=correct_opacities_t,
                scales=correct_scales_t,
                rotations=correct_rotations_t,
            )
            corrected_gaussians_masked= Gaussians(
                means=correct_means_tm,
                covariances=correct_covariances_tm,
                harmonics=correct_harmonics_tm,
                opacities=correct_opacities_tm,
                scales=correct_scales_tm,
                rotations=correct_rotations_tm,
            )

            sub_predicted = self.center_substitute_projectionbased(correct_gaussians_center, predicted_extrinsics[0], original_pixel_coordinate)
            sub_posed = self.center_substitute_projectionbased(self.centers[i], extrinsics1, original_pixel_coordinate)
            transformation = self.estimate_transformation(sub_predicted, sub_posed)
            transformed_posed=sub_predicted@transformation[0:3,0:3].T+transformation[:3,3]
            print('Posing Error: ', transformed_posed-sub_posed)

            og_ex_gaussians = self.transform_gaussians_any(corrected_gaussians_masked,transformation[0:3,0:3],transformation[:3,3])
            og_ex_gaussians_unmasked = self.transform_gaussians_any(corrected_gaussians,transformation[0:3,0:3],transformation[:3,3])
            new_rotated_corrected_center = og_ex_gaussians_unmasked.means[0].clone().cpu().detach().numpy()[(target_v1 * 2 * 448) + target_u1 * 2]
            print('new_rotated_corrected_center:',new_rotated_corrected_center)


            if visual_toggle:
                v1_gs_og, v1_gs_og_depth = self.render_splat_ret_img(means=gaussians.means,
                                                           scale=gaussians.scales,
                                                           rot=gaussians.rotations,
                                                           harmonics=gaussians.harmonics,
                                                           opacity=gaussians.opacities,
                                                           covariance=gaussians.covariances,
                                                           extrinsics=predicted_extrinsics[0],
                                                           intrinsics=predicted_intrinsics_448[0],
                                                           h=448,
                                                           w=448)
                v2_gs_og, v2_gs_og_depth = self.render_splat_ret_img(means=gaussians.means,
                                                           scale=gaussians.scales,
                                                           rot=gaussians.rotations,
                                                           harmonics=gaussians.harmonics,
                                                           opacity=gaussians.opacities,
                                                           covariance=gaussians.covariances,
                                                           extrinsics=predicted_extrinsics[1],
                                                           intrinsics=predicted_intrinsics_448[1],
                                                           h=448,
                                                           w=448)

                v1_gs_og_z, v1_gs_og_z_depth = self.render_splat_ret_img(means=means_zscale_t,
                                                           scale=scales_zscale_t,
                                                           rot=rotations_zscale_t,
                                                           harmonics=harmonics_zscale_t,
                                                           opacity=opacities_zscale_t,
                                                           covariance=covariance_zscale_t,
                                                           extrinsics=predicted_extrinsics[0],
                                                           intrinsics=predicted_intrinsics_448[0],
                                                           h=448,
                                                           w=448)
                v2_gs_og_z, v2_gs_og_z_depth = self.render_splat_ret_img(means=means_zscale_t,
                                                           scale=scales_zscale_t,
                                                           rot=rotations_zscale_t,
                                                           harmonics=harmonics_zscale_t,
                                                           opacity=opacities_zscale_t,
                                                           covariance=covariance_zscale_t,
                                                           extrinsics=predicted_extrinsics[1],
                                                           intrinsics=predicted_intrinsics_448[1],
                                                           h=448,
                                                           w=448)
                v1_gs_meshview, v1_gs_meshview_depth = self.render_splat_ret_img(means=og_ex_gaussians.means,
                                                           scale=og_ex_gaussians.scales,
                                                           rot=og_ex_gaussians.rotations,
                                                           harmonics=og_ex_gaussians.harmonics,
                                                           opacity=og_ex_gaussians.opacities,
                                                           covariance=og_ex_gaussians.covariances,
                                                           extrinsics=extrinsics1,
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v2_gs_meshview, v2_gs_meshview_depth = self.render_splat_ret_img(means=og_ex_gaussians.means,
                                                           scale=og_ex_gaussians.scales,
                                                           rot=og_ex_gaussians.rotations,
                                                           harmonics=og_ex_gaussians.harmonics,
                                                           opacity=og_ex_gaussians.opacities,
                                                           covariance=og_ex_gaussians.covariances,
                                                           extrinsics=extrinsics2,
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v1_gs_meshview_um, v1_gs_meshview_um_depth = self.render_splat_ret_img(means=og_ex_gaussians_unmasked.means,
                                                           scale=og_ex_gaussians_unmasked.scales,
                                                           rot=og_ex_gaussians_unmasked.rotations,
                                                           harmonics=og_ex_gaussians_unmasked.harmonics,
                                                           opacity=og_ex_gaussians_unmasked.opacities,
                                                           covariance=og_ex_gaussians_unmasked.covariances,
                                                           extrinsics=extrinsics1,
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v2_gs_meshview_um, v2_gs_meshview_um_depth = self.render_splat_ret_img(means=og_ex_gaussians_unmasked.means,
                                                           scale=og_ex_gaussians_unmasked.scales,
                                                           rot=og_ex_gaussians_unmasked.rotations,
                                                           harmonics=og_ex_gaussians_unmasked.harmonics,
                                                           opacity=og_ex_gaussians_unmasked.opacities,
                                                           covariance=og_ex_gaussians_unmasked.covariances,
                                                           extrinsics=extrinsics2,
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v1_gs_corrected, v1_gs_corrected_depth = self.render_splat_ret_img(means=correct_gaussians.means,
                                                           scale=correct_gaussians.scales,
                                                           rot=correct_gaussians.rotations,
                                                           harmonics=correct_gaussians.harmonics,
                                                           opacity=correct_gaussians.opacities,
                                                           covariance=correct_gaussians.covariances,
                                                           extrinsics=predicted_extrinsics[0],
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v2_gs_corrected, v2_gs_corrected_depth = self.render_splat_ret_img(means=correct_gaussians.means,
                                                           scale=correct_gaussians.scales,
                                                           rot=correct_gaussians.rotations,
                                                           harmonics=correct_gaussians.harmonics,
                                                           opacity=correct_gaussians.opacities,
                                                           covariance=correct_gaussians.covariances,
                                                           extrinsics=predicted_extrinsics[1],
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v1_gs_corrected_masked, v1_gs_corrected_masked_depth = self.render_splat_ret_img(means=correct_means_tm,
                                                           scale=correct_scales_tm,
                                                           rot=correct_rotations_tm,
                                                           harmonics=correct_harmonics_tm,
                                                           opacity=correct_opacities_tm,
                                                           covariance=correct_covariances_tm,
                                                           extrinsics=predicted_extrinsics[0],
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)
                v2_gs_corrected_masked, v2_gs_corrected_masked_depth = self.render_splat_ret_img(means=correct_means_tm,
                                                           scale=correct_scales_tm,
                                                           rot=correct_rotations_tm,
                                                           harmonics=correct_harmonics_tm,
                                                           opacity=correct_opacities_tm,
                                                           covariance=correct_covariances_tm,
                                                           extrinsics=predicted_extrinsics[1],
                                                           intrinsics=self.K,
                                                           h=self.img_h,
                                                           w=self.img_w)

                f, axarr = plt.subplots(2, 8)
                axarr[0, 0].imshow(self.images[imagenr1][crop_starty1:crop_starty1 + 224, crop_startx1:crop_startx1 + 224, :])
                axarr[0, 0].set_title('view1')
                axarr[0, 1].imshow(v1_gs_meshview[crop_starty1:crop_starty1 + 224, crop_startx1:crop_startx1 + 224, :])
                axarr[0, 1].set_title('v1_gs_meshview')
                axarr[0, 2].imshow(v1_gs_meshview_um[crop_starty1:crop_starty1 + 224, crop_startx1:crop_startx1 + 224, :])
                axarr[0, 2].set_title('v1_gs_meshview_um')
                axarr[0, 3].imshow(v1_gs_meshview)
                axarr[0, 3].set_title('v1_gs_meshview')
                axarr[0, 4].imshow(v1_gs_meshview_um)
                axarr[0, 4].set_title('v1_gs_meshview_um')
                axarr[0, 5].imshow(v1_gs_corrected[crop_starty1:crop_starty1 + 224, crop_startx1:crop_startx1 + 224, :])
                axarr[0, 5].set_title('v1_gs_corrected')
                axarr[0, 6].imshow(v1_gs_corrected_masked[crop_starty1:crop_starty1 + 224, crop_startx1:crop_startx1 + 224, :])
                axarr[0, 6].set_title('v1_gs_corrected_masked')
                axarr[0, 7].imshow(v1_gs_og)
                axarr[0, 7].set_title('v1_gs_og')
                axarr[1, 0].imshow(self.images[imagenr2][crop_starty2:crop_starty2 + 224, crop_startx2:crop_startx2 + 224, :])
                axarr[1, 0].set_title('view2')
                axarr[1, 1].imshow(v2_gs_meshview[crop_starty2:crop_starty2 + 224, crop_startx2:crop_startx2 + 224, :])
                axarr[1, 1].set_title('v2_gs_meshview')
                axarr[1, 2].imshow(v2_gs_meshview_um[crop_starty2:crop_starty2 + 224, crop_startx2:crop_startx2 + 224, :])
                axarr[1, 2].set_title('v2_gs_meshview_um')
                axarr[1, 3].imshow(v2_gs_meshview)
                axarr[1, 3].set_title('v2_gs_meshview')
                axarr[1, 4].imshow(v2_gs_meshview_um)
                axarr[1, 4].set_title('v2_gs_meshview_um')
                axarr[1, 5].imshow(v2_gs_corrected[crop_starty2:crop_starty2 + 224, crop_startx2:crop_startx2 + 224, :])
                axarr[1, 5].set_title('v2_gs_corrected')
                axarr[1, 6].imshow(v2_gs_corrected_masked[crop_starty2:crop_starty2 + 224, crop_startx2:crop_startx2 + 224, :])
                axarr[1, 6].set_title('v2_gs_corrected_masked')
                axarr[1, 7].imshow(v2_gs_og)
                axarr[1, 7].set_title('v2_gs_og')
                plt.show()
                time.sleep(1)
            #export_ply(og_ex_gaussians.means[0], og_ex_gaussians.scales[0], og_ex_gaussians.rotations[0], og_ex_gaussians.harmonics[0],
            #           og_ex_gaussians.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians'+str(self.vol_radius)+'m2picsoptim') / ("gaussiansoptim"+str(i)+".ply"))
            export_ply(og_ex_gaussians_unmasked.means[0], og_ex_gaussians_unmasked.scales[0], og_ex_gaussians_unmasked.rotations[0], og_ex_gaussians_unmasked.harmonics[0],
                       og_ex_gaussians_unmasked.opacities[0], Path('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians'+str(self.vol_radius)+'m2picsoptim') / ("gaussiansoptim"+str(i)+".ply"))
        return

    import numpy as np

    def estimate_transformation(self, source_points, target_points):
        """
        Estimate a rigid 3D transformation from corresponding points.

        Parameters
        ----------
        source_points : (N, 3) ndarray
            Points in the initial position.
        target_points : (N, 3) ndarray
            Corresponding points in the target position.

        Returns
        -------
        T : (4, 4) ndarray
            Homogeneous transformation matrix.
        """
        source_points = np.asarray(source_points, dtype=float)
        target_points = np.asarray(target_points, dtype=float)

        assert source_points.shape == target_points.shape
        assert source_points.ndim == 2 and source_points.shape[1] == 3
        assert len(source_points) >= 3

        # Calculate centroids
        centroid_source = source_points.mean(axis=0)
        centroid_target = target_points.mean(axis=0)

        # Center the points
        P = source_points - centroid_source
        Q = target_points - centroid_target

        # Compute covariance matrix
        H = P.T @ Q

        # Singular value decomposition
        U, S, Vt = np.linalg.svd(H)

        # Calculate rotation
        R = Vt.T @ U.T

        # Ensure a proper rotation (det(R) = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Calculate translation
        t = centroid_target - R @ centroid_source

        # Construct homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def rotation_matrix_from_vectors(self,vec_a, vec_b):
        """Find the rotation matrix that aligns vec_a to vec_b."""
        a = vec_a / np.linalg.norm(vec_a)
        b = vec_b / np.linalg.norm(vec_b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)

        if s == 0:
            return np.identity(3)  # Vectors are identical or opposite

        # Skew-symmetric cross-product matrix of v
        v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # Rodrigues' rotation formula
        r = np.identity(3) + v_x + (v_x @ v_x) * ((1 - c) / (s ** 2))
        return r
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


    def center_substitute_projectionbased(self,centervert, extrinsics, targetpix):
        rotangle = np.pi / 2
        rx = np.array([[1, 0, 0], [0, np.cos(rotangle), -np.sin(rotangle)], [0, np.sin(rotangle), np.cos(rotangle)]])
        ry = np.array([[np.cos(rotangle), 0, np.sin(rotangle)], [0, 1, 0], [-np.sin(rotangle), 0, np.cos(rotangle)]])
        rz = np.array([[np.cos(rotangle), -np.sin(rotangle), 0], [np.sin(rotangle), np.cos(rotangle), 0], [0, 0, 1]])

        origin_a, direction_a = self.pixel_to_world_ray(np.array([targetpix[0], targetpix[1]]), self.K,extrinsics)
        origin_b, direction_b = self.pixel_to_world_ray(np.array([0, 0]), self.K,extrinsics)

        direction_ax=rx@direction_a
        direction_ay=ry@direction_a
        direction_az=rz@direction_a

        #xyzcords=np.array([centervert + direction_a,centervert -direction_a,centervert + direction_ax,centervert -direction_ax,centervert + direction_ay,centervert -direction_ay,centervert + direction_az,centervert -direction_az,centervert])
        #xyzcords=np.array([centervert + direction_a,centervert -direction_a,centervert])
        xyzcords=np.array([centervert,centervert + direction_a,centervert -direction_a,centervert + direction_b,centervert -direction_b])

        return xyzcords

    def submap_gs_stitching_pipeline(self):
        for key in self.subcenter_dict.keys():
            #if key == 0: continue
        #for j in range(len(self.centers)):
            j=key
            path = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussians"+str(self.vol_radius)+"m2picsoptim/gaussiansoptim" + str(
                j) + ".ply"
            ply = PlyData.read(path)

            vetices = ply["vertex"].data
            rotated_verts =vetices
            if j == 0:
                corrected_verts = rotated_verts
            else:
                corrected_verts = np.concatenate([corrected_verts, rotated_verts])
            print(j)

        ply_out = PlyData([
            PlyElement.describe(corrected_verts, "vertex")
        ], text=False)
        ply_out.write(
            "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussiansubmapstitch"+str(self.vol_radius)+"mmultic_variabledscalev4.ply")
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

    def discretize_pointcloud(self):
        """
        Partition a point cloud into spherical regions centered on
        selected points. Regions may overlap.

        Parameters
        ----------
        points : ndarray, shape (N, 3)
            Input point cloud.
        radius : float
            Radius of each spherical region.

        Returns
        -------
        centers : ndarray, shape (K, 3)
            Selected centers.
        volumes : list of ndarrays
            Points belonging to each spherical region.
        """
        points= np.copy(self.pcd)
        radius = self.vol_radius
        centers = []
        volumes = []
        remaining = np.ones(len(points), dtype=bool)

        while np.any(remaining):
            # Select the first remaining point as the next center
            center_idx = np.flatnonzero(remaining)[0]
            center = points[center_idx]

            # Find all points within radius of this center
            distances = np.linalg.norm(points - center, axis=1)
            mask = distances <= radius

            centers.append(center)
            volumes.append(points[mask])

            # Mark points inside this sphere as assigned
            remaining[mask] = False

        return np.array(centers), volumes

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


    def transform_gaussians_any(self,gaussians,R,translation):

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
            covariances=build_covariance(gaussians.scales,rotations_tensor),
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=gaussians.harmonics,
            opacities=gaussians.opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=gaussians.scales,
            rotations=rotations_tensor
        )
        return newgaussians

    def transform_gaussians_individual(
            self,
            gaussians,
            extrinsics,
            ctr_img_nr,
            cropstarts,
            distance,
            targetpixel,
    ):
        H = W = 448
        S = 224

        # --------------------------------------------------
        # 1. Extract Gaussian parameters
        # --------------------------------------------------
        means = (gaussians.means[0].detach().cpu().numpy().reshape(H, W, 3).copy())

        scales = (gaussians.scales[0].detach().cpu().numpy()[:H * W].copy())

        rotations = (gaussians.rotations[0].detach().cpu().numpy().reshape(H, W, 4).copy())

        harmonics = (gaussians.harmonics[0].detach().cpu().numpy()[:H * W])

        opacities = (gaussians.opacities[0].detach().cpu().numpy()[:H * W])

        # --------------------------------------------------
        # 2. Compute ray directions
        # --------------------------------------------------
        directions_a = np.empty((S, S, 3), dtype=np.float64)
        directions_b = np.empty((S, S, 3), dtype=np.float64)

        origin_b = None

        for v in range(S):
            for u in range(S):
                pixel_a = np.array([
                    self.K[0, 2] - 112 + u,
                    self.K[1, 2] - 112 + v,
                ])

                pixel_b = np.array([
                    cropstarts[0] + u,
                    cropstarts[1] + v,
                ])

                origin_a, directions_a[v, u] = self.pixel_to_world_ray(
                    pixel_a, self.K, extrinsics
                )

                origin_b, directions_b[v, u] = self.pixel_to_world_ray(
                    pixel_b, self.K, extrinsics
                )

        # --------------------------------------------------
        # 3. Compute all rotation matrices in a batch
        # --------------------------------------------------
        # Equivalent to:
        # inv(rotation_matrix_from_vectors(direction_b,
        #                                  direction_a))
        #
        # Assuming rotation_matrix_from_vectors(a, b)
        # maps a onto b.

        a = directions_a
        b = directions_b

        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)

        cross = np.cross(a, b)
        dot = np.sum(a * b, axis=-1)

        # Skew-symmetric matrices for all pixel rotations
        K = np.zeros((S, S, 3, 3), dtype=np.float64)

        K[..., 0, 1] = -cross[..., 2]
        K[..., 0, 2] = cross[..., 1]
        K[..., 1, 0] = cross[..., 2]
        K[..., 1, 2] = -cross[..., 0]
        K[..., 2, 0] = -cross[..., 1]
        K[..., 2, 1] = cross[..., 0]

        I = np.eye(3)
        K2 = K @ K

        denominator = 1.0 + dot

        # The rays should not be antiparallel for this application.
        if np.any(denominator < 1e-8):
            raise ValueError("Antiparallel rays encountered; handle these rotations separately.")

        R = (I+ K+ K2 / denominator[..., None, None])

        # --------------------------------------------------
        # 4. Apply each rotation to its 2x2 Gaussian block
        # --------------------------------------------------
        # Block layout: (224, 224, 2, 2, 3)
        mean_blocks = (means.reshape(S, 2, S, 2, 3).transpose(0, 2, 1, 3, 4))

        # Apply R to all four points per pixel
        mean_blocks = np.einsum("uvij,uvabj->uvabi",R,mean_blocks,)

        means = (mean_blocks.transpose(0, 2, 1, 3, 4).reshape(H, W, 3))

        # --------------------------------------------------
        # 5. Apply rotations to Gaussian quaternions
        # --------------------------------------------------
        # SciPy uses (x, y, z, w).
        q_original = (rotations.reshape(S, 2, S, 2, 4).transpose(0, 2, 1, 3, 4))

        q_transform = Rotation.from_matrix(R.reshape(-1, 3, 3))
        q_transform = q_transform.as_quat().reshape(S, S, 4)

        # Broadcast the rotation quaternion to all four
        # Gaussians belonging to each pixel.
        q_transform = q_transform[:, :, None, None, :]

        qx, qy, qz, qw = np.moveaxis(q_transform, -1, 0)
        px, py, pz, pw = np.moveaxis(q_original, -1, 0)

        # Hamilton product: q_transform * q_original
        q_rotated = np.stack([
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
            qw * pw - qx * px - qy * py - qz * pz,
        ], axis=-1)

        rotations = (q_rotated.transpose(0, 2, 1, 3, 4).reshape(H, W, 4))

        # --------------------------------------------------
        # 6. Scale the scene around origin_b
        # --------------------------------------------------
        target_x = int(targetpixel[0]) * 2
        target_y = int(targetpixel[1]) * 2

        centerpt = means[target_y, target_x].copy()

        dist = np.linalg.norm(centerpt - origin_b)

        if dist < 1e-12:
            raise ValueError("Target Gaussian coincides with the camera origin.")

        distscale = distance / dist

        means = ((means.reshape(-1, 3) - origin_b) * distscale+ origin_b)
        means_inshape = means.copy().reshape(H, W, 3)
        print('distanceerror main center:',np.linalg.norm(means[target_y * W + target_x]- extrinsics[0:3, 3:].reshape((3,)))-distance)

        scales = scales * distscale

        # --------------------------------------
        # 6.5. Experiments regarding multidistance scaling
        # --------------------------------------
        subcenters_dict = self.subcenter_dict[ctr_img_nr[0]]
        n_depths=np.array([1])
        n_coords=np.array([target_x,target_y])
        maxx = 0
        maxx_dist = 0
        maxx_key =''
        og_vertices = np.array(self.centers[ctr_img_nr[0]])
        pred_vertices = np.array(self.centers[ctr_img_nr[1]])
        for key in subcenters_dict.keys():
            if 'subcenter' in str(key):
                subcenter = subcenters_dict[key]
                pixel = subcenter['pixel']
                nr = subcenter['centernumber']
                vertex = subcenter['vert']
                targetx_sub = (pixel[0]-cropstarts[0])*2
                xdist = abs(targetx_sub-target_x)
                if xdist > maxx:
                    maxx_key=str(key)
                    maxx_dist=targetx_sub
                    maxx=xdist
                    targety_sub = (pixel[1] - cropstarts[1]) * 2
                    n_coords = np.vstack((n_coords, [targetx_sub, targety_sub]))
                    distance_subc = np.linalg.norm(vertex - self.poses[ctr_img_nr[1]][0:3, 3:].reshape((3,)))
                    distance_subc_pred = np.linalg.norm(
                        means[targety_sub * W + targetx_sub] - extrinsics[0:3, 3:].reshape((3,)))
                    n_depths = np.concatenate((n_depths, np.array([distance_subc_pred / distance_subc])), axis=None)
                else:
                    continue

        if maxx!=0:
            depth_correctionmap = np.zeros([448, 448])
            scalar=1
            if maxx<target_x:
                smaller = maxx
                bigger=target_x
                smaller_dscale = n_depths[-1]
                bigger_dscale=1
            else:
                smaller = target_x
                bigger = maxx
                smaller_dscale = 1
                bigger_dscale=n_depths[-1]

            for i in range(224):
                current = i*2
                if current <= smaller:
                    scalar= smaller_dscale
                elif current >= bigger:
                    scalar= bigger_dscale
                else:
                    maxdist=bigger-smaller
                    d2small = current - smaller
                    d2big = bigger - current
                    scalar=((1-d2small/maxdist)*smaller_dscale)+((1-d2big/maxdist)*bigger_dscale)
                depth_correctionmap[:,current:current+2]+=scalar
            #depth_correctionmap = self.create_depth_map(n_coords,n_depths,size=448,sigma=10.0)
            #depth_correctionmap = self.create_smooth_depth_map_rbf(n_coords,n_depths)
            depth_correctionmap = np.clip(depth_correctionmap,0.8,1.2)
            means_corrected = depth_correctionmap[:, :, np.newaxis] * means_inshape
            means = means_corrected.reshape(-1, 3)
            scales = depth_correctionmap[:, :, np.newaxis]*scales.reshape(448,448,3)
            scales = scales.reshape(448*448,3)
        # --------------------------------------------------
        # 7. Convert back to PyTorch tensors
        # --------------------------------------------------
        device = gaussians.means.device
        dtype = gaussians.means.dtype

        means_t = torch.as_tensor(means, device=device, dtype=dtype).unsqueeze(0)

        scales_t = torch.as_tensor(scales, device=gaussians.scales.device,dtype=gaussians.scales.dtype).unsqueeze(0)

        rotations_t = torch.as_tensor(rotations.reshape(-1, 4),device=gaussians.rotations.device,dtype=gaussians.rotations.dtype).unsqueeze(0)

        harmonics_t = torch.as_tensor(harmonics, device=gaussians.harmonics.device,dtype=gaussians.harmonics.dtype).unsqueeze(0)

        opacities_t = torch.as_tensor(opacities, device=gaussians.opacities.device,dtype=gaussians.opacities.dtype).unsqueeze(0)

        covariances_t = build_covariance(scales_t, rotations_t)

        newgaussians = Gaussians(
            means=means_t,
            covariances=covariances_t,
            harmonics=harmonics_t,
            opacities=opacities_t,
            scales=scales_t,
            rotations=rotations_t,
        )

        target_mean = means[target_y * W + target_x]

        return newgaussians, target_mean

    def create_smooth_depth_map_rbf(self,pixel_coords, depths, size=448):
        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64).reshape(-1)

        # Generate pixel coordinates.
        x, y = np.meshgrid(
            np.arange(size),
            np.arange(size)
        )

        grid_points = np.column_stack((x.ravel(), y.ravel()))

        # Fit a smooth interpolator.
        interpolator = RBFInterpolator(
            pixel_coords,
            depths,
            kernel="thin_plate_spline",
            smoothing=0.0
        )

        depth_map = interpolator(grid_points)

        return depth_map.reshape(size, size)
    def create_depth_map(self,pixel_coords, depths, size=448, sigma=2.0):
        """
        Create a smooth depth map from sparse depth measurements.

        Parameters
        ----------
        pixel_coords : np.ndarray, shape (N, 2)
            Pixel coordinates in (x, y) format.
        depths : np.ndarray, shape (N,)
            Depth value for each pixel coordinate.
        size : int
            Width and height of the output depth map.
        sigma : float
            Gaussian smoothing strength in pixels.

        Returns
        -------
        depth_map : np.ndarray, shape (size, size)
            Smooth depth map. Unobserved regions outside the
            convex hull are filled using nearest-neighbor interpolation.
        """

        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64).reshape(-1)

        if len(pixel_coords) != len(depths):
            raise ValueError("Each pixel coordinate must have a depth value.")

        if len(depths) == 0:
            raise ValueError("At least one depth measurement is required.")

        # Create the pixel grid.
        x, y = np.meshgrid(
            np.arange(size),
            np.arange(size)
        )

        grid_points = np.column_stack((x.ravel(), y.ravel()))

        # Linear interpolation.
        depth_map = griddata(
            pixel_coords,
            depths,
            grid_points,
            method="linear"
        ).reshape(size, size)

        # Fill areas outside the convex hull.
        missing = np.isnan(depth_map)

        if np.any(missing):
            nearest_map = griddata(
                pixel_coords,
                depths,
                grid_points,
                method="nearest"
            ).reshape(size, size)

            depth_map[missing] = nearest_map[missing]

        # Smooth the resulting depth map.
        depth_map = gaussian_filter(depth_map, sigma=sigma)

        return depth_map

    def transform_gaussians_individual2(self,gaussians,extrinsics,intrinsics, cropstarts, distance, targetpixel):

        means = gaussians.means[0].clone().cpu().detach().numpy()
        means=means.reshape((1,448,448,3))
        scales = gaussians.scales[0].clone().cpu().detach().numpy()[0:448*448]
        rotations = gaussians.rotations[0].clone().cpu().detach().numpy()
        rotations = rotations.reshape((1, 448, 448, 4))
        harmonics = gaussians.harmonics[0].clone().cpu().detach().numpy()[0:448*448]
        opacities = gaussians.opacities[0].clone().cpu().detach().numpy()[0:448*448]


        for u in range(224):
            for v in range(224):


                origin_a, direction_a = self.pixel_to_world_ray(np.array([self.K[0, 2]-112+u, self.K[1, 2]-112+v]),self.K, extrinsics)
                origin_b, direction_b = self.pixel_to_world_ray(np.array([cropstarts[0]+u, cropstarts[1]+v]),self.K, extrinsics)

                r = np.linalg.inv(self.rotation_matrix_from_vectors(direction_b, direction_a))
                means[0,v*2,u*2,:]=r@means[0,v*2,u*2,:]
                means[0,v*2+1,u*2,:]=r@means[0,v*2+1,u*2,:]
                means[0,v*2,u*2+1,:]=r@means[0,v*2,u*2+1,:]
                means[0,v*2+1,u*2+1,:]=r@means[0,v*2+1,u*2+1,:]
                r = Rotation.from_matrix(r)

                rotations[0,v*2,u*2,:]=(r *Rotation.from_quat(rotations[0,v*2,u*2,:])).as_quat()
                rotations[0,v*2+1,u*2,:]=(r *Rotation.from_quat(rotations[0,v*2+1,u*2,:])).as_quat()
                rotations[0,v*2,u*2+1,:]=(r *Rotation.from_quat(rotations[0,v*2,u*2+1,:])).as_quat()
                rotations[0,v*2+1,u*2+1,:]=(r *Rotation.from_quat(rotations[0,v*2+1,u*2+1,:])).as_quat()

        centerpt=means[0,targetpixel[1]*2,targetpixel[0]*2,:]
        dist = np.linalg.norm(centerpt-origin_b)
        distscale=distance/dist
        means=((means.reshape((448*448,3))[0:448*448]-np.array(origin_b))*distscale)+np.array(origin_b)
        means_t = torch.tensor(means, dtype=torch.float32, device="cuda")[None]
        scales_t = torch.tensor(scales*distscale, dtype=torch.float32, device="cuda")[None]
        rotations_t = torch.tensor(rotations.reshape((448*448,4))[0:448*448], dtype=torch.float32, device="cuda")[None]
        harmonics_t = torch.tensor(harmonics, dtype=torch.float32, device="cuda")[None]
        opacities_t = torch.tensor(opacities, dtype=torch.float32, device="cuda")[None]
        covariances_t=build_covariance(scales_t, rotations_t)
        '''v1_gs_og, v1_gs_og_depth = self.render_splat_ret_img(means=gaussians.means,
                                                   scale=gaussians.scales,
                                                   rot=gaussians.rotations,
                                                   harmonics=gaussians.harmonics,
                                                   opacity=gaussians.opacities,
                                                   covariance=gaussians.covariances,
                                                   extrinsics=extrinsics,
                                                   intrinsics=intrinsics,
                                                   h=224,
                                                   w=224)

        v1_gs_reduced, v1_gs_reduced_depth = self.render_splat_ret_img(means=means_t,
                                                   scale=scales_t,
                                                   rot=rotations_t,
                                                   harmonics=harmonics_t,
                                                   opacity=opacities_t,
                                                   covariance=covariances_t,
                                                   extrinsics=extrinsics,
                                                   intrinsics=self.K,
                                                   h=self.img_h,
                                                   w=self.img_w)
        f, axarr = plt.subplots(3)
        axarr[0].imshow(v1_gs_og)
        axarr[0].set_title('v1_gs_og')
        axarr[1].imshow(v1_gs_reduced)
        axarr[1].set_title('v1_gs_reduced')
        axarr[2].imshow(v1_gs_reduced[cropstarts[1]:cropstarts[1] + 224, cropstarts[0]:cropstarts[0] + 224, :])
        axarr[2].set_title('v1_gs_reducedcropped')
        plt.show()
        time.sleep(1)'''
        newgaussians = Gaussians(
            means=means_t,
            covariances=covariances_t,
            harmonics=harmonics_t,
            opacities=opacities_t,
            scales=scales_t,
            rotations=rotations_t
        )

        return newgaussians,means[(targetpixel[1]*2 * 448) + targetpixel[0]*2]


def main():
    meshpath = "/home/luis/Desktop/SABERGUIDE/KITTI/kitti_seq10_gt/mesh_kf000001.ply"
    framestart=0
    frameend=34
    submap = Submap(Config, framestart, frameend)
    submap.init_geometry(meshpath)
    submap.init_model()
    submap.init_model_inputs()
    #submap.pipeline_foward(visual_toggle=True)
    submap.pipeline_foward()
    submap.submap_gs_stitching_pipeline()
    return


if __name__ == '__main__':
    sys.exit(main())