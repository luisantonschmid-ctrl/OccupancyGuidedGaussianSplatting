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

NEAR_CLIP = 0.5  # metres — discard points closer than this
FAR_CLIP = 160.0  # metres — discard points farther than this
POINT_SIZE = 2  # blob radius in pixels (1 = single pixel)
OVERLAY_ALPHA = 0.6  # mesh weight in the blended overlay image

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
    def __init__(self,config,frame_start,frame_end, model,rot_arr,mesh_path):
        self.submapname=mesh_path.split('/')[-1]
        self.config = config
        self.model=model
        self.rot_arr=rot_arr
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
        self.vol_radius = config.vol_radius
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vol_origins_extracted = []
        self.keylist =[]
        self.poses=self.load_poses(Path(self.config.pose_path))[self.frame_start:self.frame_end+1,:,:]
        self.mesh = trimesh.load_mesh(mesh_path)
        self.images = self.fetch_images()

        self.start = time.perf_counter()
        intrinsicsfxy = self.K[0,0]
        self.model.encoder.intrinsicsfxy = intrinsicsfxy

    def init_geometry(self):
        self.maxlen=len(self.load_poses(Path(self.config.pose_path)))
        self.pcd = self.mesh.triangles_center.astype(np.float32)
        if not self.frame_end >= self.maxlen - 2:
            nextscenemask = self.return_nextscenemask(self.pcd)
            self.pcd=self.pcd[nextscenemask]

        self.centers, self.volumes = self.discretize_pointcloud()
        self.inverted_poses = self.invert_poses()
        projection_results = [
            self.project_centers(i)
            for i in range(len(self.images))
        ]

        self.im_u = np.stack(
            [result[0] for result in projection_results],
            axis=0
        )

        self.im_v = np.stack(
            [result[1] for result in projection_results],
            axis=0
        )

        self.inside_mask = np.stack(
            [result[2] for result in projection_results],
            axis=0
        )
        return

    def init_model_inputs(self):
        self.crop_dict = {}
        self.center_shiftdict = {}
        self.crop_starts_global = {}
        self.subcenter_dict = {}
        for j in range(len(self.centers)):
            flag=0
            for l in range(self.frame_end-self.frame_start):
                k=self.frame_end-self.frame_start-l-1
                if self.inside_mask[k][j]:
                    flag+=1
                    #print(self.im_u[k][j], self.im_v[k][j])
                    crop, centershift, start_points = self.extract_window_no_padding(self.images[k], int(self.im_u[k][j]),
                                                                         int(self.im_v[k][j]), plot=False)
                    self.center_shiftdict[str("center" + str(j) + "image" + str(k))] = {"x_shift": str(centershift[0]),
                                                                                        "y_shift": str(centershift[1])}
                    self.crop_starts_global[str("center" + str(j) + "image" + str(k))] = {"x": str(start_points[0]),
                                                                                   "y": str(start_points[1])}
                    self.crop_dict[str("center" + str(j) + "image" + str(k))] = {"crop": crop}
                    if flag>=2:
                        break
                    #cv2.imwrite('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/mini-splatting2/output/cropped2m/center' + str(j) + 'image' + str(k) + '.png', crop)
        iterator = np.arange(len(self.centers))
        iterator_subc = np.arange(len(self.centers))

        while iterator.size != 0:
            centernumber=iterator[0]
            iterator = iterator[1:]
            if np.sum(self.inside_mask[:,centernumber]) == 0:
                continue
            crop_path = list(filter(lambda x: x.startswith("center" + str(centernumber) + "image"), sorted(list(self.center_shiftdict.keys()))))[-1]
            imagenr = int(crop_path.removeprefix("center" + str(centernumber) + "image"))
            u,v=self.im_u[imagenr][centernumber],self.im_v[imagenr][centernumber]
            centervert = self.centers[centernumber]
            crop = self.crop_dict[crop_path]['crop']
            crop_start = self.crop_starts_global[crop_path]
            self.subcenter_dict[centernumber] = {'pixel':[u,v], 'vert':centervert,'crop':crop,'crop_start':crop_start,'imagenr':imagenr}
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
        self.overlap_dict = self.overlap_masks()

        return

    def transform_world(self, key, predex):
        invex=np.linalg.inv(predex)
        self.transform_gaussians_any(key,invex[0:3,0:3],invex[0:3,3])
        self.subcenter_dict[key]['means'][:, 2] = self.subcenter_dict[key]['means'][:, 2] * 2
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

    def pipeline_foward(self, visual_toggle=False):
        device = self.device
        max_batch_size = 15

        # --------------------------------------------------
        # 1. Index crop paths once instead of sorting/filtering
        #    all dictionary keys for every center.
        # --------------------------------------------------
        crop_paths_by_center = {}

        for path in self.center_shiftdict:
            # Expected format: center{key}image{frame}
            center, frame = path.split("image", 1)
            key = int(center.removeprefix("center"))

            crop_paths_by_center.setdefault(key, []).append(
                (int(frame), path)
            )

        # Sort only the paths belonging to each center.
        for paths in crop_paths_by_center.values():
            paths.sort(key=lambda item: item[0])

        # --------------------------------------------------
        # 2. Prepare batches
        # --------------------------------------------------
        all_keys = []
        image_batches = []

        for key in self.subcenter_dict:
            paths = crop_paths_by_center.get(key)

            if not paths:
                continue

            first_path = paths[-1][1]
            last_path = paths[0][1]

            # Preserve the original preprocessing and normalization.
            images = torch.stack(
                [
                    process_image_helper(
                        self.crop_dict[first_path]["crop"]
                    ),
                    process_image_helper(
                        self.crop_dict[last_path]["crop"]
                    ),
                ],
                dim=0,
            )

            all_keys.append(key)
            image_batches.append(images)

            if len(image_batches) == max_batch_size:
                batch = torch.stack(image_batches, dim=0)
                yield_batch = batch.to(device, non_blocking=True)
                yield_batch = (yield_batch + 1.0) * 0.5

                yield_batch_keys = all_keys[-len(image_batches):]

                # Run inference immediately.
                self._infer_and_store(yield_batch, yield_batch_keys)

                image_batches.clear()

        # Process the final, possibly smaller batch.
        if image_batches:
            batch = torch.stack(image_batches, dim=0)
            batch = batch.to(device, non_blocking=True)
            batch = (batch + 1.0) * 0.5

            self._infer_and_store(batch, all_keys[-len(image_batches):])

        # Store the keys for the subsequent processing stage.
        self.keylist = list(self.subcenter_dict.keys())

        # Process centers after inference.
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(
                self.forward_perkey,
                range(len(self.keylist))
            ))
        return

    def _infer_and_store(self, images, keys):
        """Run AnySplat inference and store its output for each center."""

        self.model.eval()

        with torch.inference_mode():
            gaussians, pred_context_pose = self.model.inference(images)

        # Transfer only the required data to CPU.
        means = gaussians.means.detach().cpu().numpy()
        scales = gaussians.scales.detach().cpu().numpy()
        rotations = gaussians.rotations.detach().cpu().numpy()
        harmonics = gaussians.harmonics.detach().cpu().numpy()
        opacities = gaussians.opacities.detach().cpu().numpy()
        covariances = gaussians.covariances.detach().cpu().numpy()

        predicted_extrinsics = (
            pred_context_pose["extrinsic"].detach().cpu().numpy()
        )
        predicted_intrinsics = (
            pred_context_pose["intrinsic"].detach().cpu().numpy()
        )

        n_gaussians = 448 * 448

        for j, key in enumerate(keys):
            data = self.subcenter_dict[key]

            data["means"] = means[j, :n_gaussians]
            data["scales"] = scales[j, :n_gaussians]
            data["rotations"] = rotations[j, :n_gaussians]
            data["harmonics"] = harmonics[j, :n_gaussians]
            data["opacities"] = opacities[j, :n_gaussians]
            data["covariances"] = covariances[j, :n_gaussians]

            data["pred_ex"] = predicted_extrinsics[j]
            data["pred_intr"] = predicted_intrinsics[j]
    def forward_perkey(self, k):
        key = self.keylist[k]

        print('processing center: ', key)
        # nr_keys += 1
        # index_gaussianlist = math.floor()
        # for i in range(len(self.centers)):
        i = key
        crop_paths = list(
            filter(lambda x: x.startswith("center" + str(i) + "image"), sorted(list(self.center_shiftdict.keys()))))
        crop_paths = [crop_paths[-1], crop_paths[0]]
        # print(crop_paths)
        imagenr1 = int(crop_paths[0].removeprefix("center" + str(i) + "image"))

        images = [process_image_helper(self.crop_dict[crop_paths[0]]["crop"]),
                  process_image_helper(self.crop_dict[crop_paths[-1]]["crop"])]
        images = torch.stack(images, dim=0).unsqueeze(0).to(self.device)  # [1, K, 3, 448, 448]
        b, v, _, h, w = images.shape

        crop_startx1 = int(self.crop_starts_global[crop_paths[0]]['x'])
        crop_starty1 = int(self.crop_starts_global[crop_paths[0]]['y'])


        target_u1, target_v1 = 112 - int(self.center_shiftdict[crop_paths[0]]['x_shift']), 112 - int(
            self.center_shiftdict[crop_paths[0]]['y_shift'])

        plot = True
        if plot:
            image1 = self.crop_dict[crop_paths[0]]["crop"].copy()
            image2 = self.crop_dict[crop_paths[-1]]["crop"].copy()
            image1[target_v1][target_u1] = [255, 0, 0]
            try:
                image1[target_v1 + 1][target_u1] = [255, 0, 0]
                image1[target_v1 - 1][target_u1] = [255, 0, 0]
                image1[target_v1][target_u1 + 1] = [255, 0, 0]
                image1[target_v1][target_u1 - 1] = [255, 0, 0]
            except:
                print('couldnt fill all surrounding pixels')
            # f, axarr = plt.subplots(2, 1)
            # axarr[0].imshow(image1)
            # axarr[1].imshow(image2)
            # cv2.imwrite('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/cropswithcenter2m/center' + str(i) +'image'+ str(imagenr1+self.frame_start) + '.png', image1)
            # cv2.imwrite('/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/cropswithcenter2m/center' + str(i) +'image2'+ '.png', image2)
            # plt.show()
        distance1 = np.linalg.norm(self.centers[i] - self.poses[imagenr1][0:3, 3:].reshape((3,)))
        # print('Distance '+str(imagenr1 + self.frame_start) ,distance1,self.centers[i],self.poses[imagenr1][0:3,3:].reshape((3,)))
        # print('Distance '+str(imagenr2 + self.frame_start) ,distance2,self.centers[i],self.poses[imagenr2][0:3,3:].reshape((3,)))
        # print(self.im_u[imagenr1][i],self.im_v[imagenr1][i])
        # self.model.encoder.pixeldepth1 = (target_u1,target_v1,distance1)
        # self.model.encoder.pixeldepth2 = (target_u2,target_v2,distance1)

        extrinsics1 = self.poses[imagenr1]
        extrinsics_inv1 = self.inverted_poses[imagenr1]

        predicted_extrinsics = self.subcenter_dict[key]['pred_ex']
        predicted_intrinsics = self.subcenter_dict[key]['pred_intr']
        predicted_intrinsics_448 = predicted_intrinsics.copy()
        predicted_intrinsics_448[:, 0:2, :] = predicted_intrinsics_448[:, 0:2, :] * 448
        zeroextrinsics = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        cutom_intr = predicted_intrinsics_448.copy()
        cutom_intr[0, 0, 2] = 112
        cutom_intr[0, 1, 2] = 112
        cutom_intr[1, 0, 2] = 112
        cutom_intr[1, 1, 2] = 112

        self.transform_world(key, predicted_extrinsics[0])

        correct_gaussians_center, pred_verts, og_verts = self.transform_gaussians_individual(key, zeroextrinsics,
                                                                                             (i, imagenr1),
                                                                                             (crop_startx1,
                                                                                              crop_starty1), distance1,
                                                                                             (target_u1, target_v1))

        transformation = self.estimate_transformation(pred_verts, og_verts)

        self.transform_gaussians_any(key, transformation[0:3, 0:3], transformation[:3, 3])

        '''v1_gs_meshview_um, v1_gs_meshview_um_depth = self.render_splat_ret_img(means=og_ex_gaussians_unmasked.means,
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

        twoimg = np.zeros((224,448,3))
        twoimg[:,:224,:]=image1
        twoimg[:, 224:, :] =v1_gs_meshview_um[crop_starty1:crop_starty1 + 224, crop_startx1:crop_startx1 + 224, :]

        cv2.imwrite(
            '/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/inputoutput/image' + str(imagenr1 + self.frame_start) + 'center' + str(
                i)+ '.png', twoimg)'''
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
            elif (i < 1000) and (i > 99):
                imges.append(
                    self.config.image_path + "/000" + str(i) + ".png")
            else:
                imges.append(
                    self.config.image_path + "/00" + str(i) + ".png")
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
        T = self.inverted_poses[image_nr]
        R = T[:3, :3]
        t = T[:3, 3]

        # Equivalent to multiplying homogeneous coordinates by T.
        pts_c = self.centers @ R.T + t

        X = pts_c[:, 0]
        Y = pts_c[:, 1]
        Z = pts_c[:, 2]

        valid = (Z > NEAR_CLIP) & (Z < FAR_CLIP)

        # Avoid division by zero for points with invalid depth.
        safe_Z = np.where(valid, Z, 1.0)

        u = (
                self.K[0, 0] * X / safe_Z + self.K[0, 2]
        ).astype(np.int32)

        v = (
                self.K[1, 1] * Y / safe_Z + self.K[1, 2]
        ).astype(np.int32)

        in_img = (
                (u >= 0) & (u < self.img_w) &
                (v >= 0) & (v < self.img_h)
        )

        return u, v, valid & in_img

    def project_points(self, image_nr, points, cropstartx, cropstarty):

        N = len(points)

        pts_c = (
                self.inverted_poses[image_nr]
                @ np.hstack([points, np.ones((N, 1))]).T
        ).T

        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

        valid = (Z > NEAR_CLIP) & (Z < FAR_CLIP)

        u = (self.K[0, 0] * X / Z + self.K[0, 2]).astype(np.int32)
        v = (self.K[1, 1] * Y / Z + self.K[1, 2]).astype(np.int32)

        in_img = (
                (u >= cropstartx) & (u < cropstartx + 224)
                & (v >= cropstarty) & (v < cropstarty + 224)
        )

        exclusionmask = in_img & valid

        pixels = np.column_stack((u, v))

        # Keep only the closest point for each pixel coordinate.
        valid_indices = np.flatnonzero(exclusionmask)

        if len(valid_indices) > 0:
            valid_pixels = pixels[valid_indices]

            # Group points by pixel coordinate.
            _, inverse = np.unique(
                valid_pixels,
                axis=0,
                return_inverse=True
            )

            # Find the closest point (smallest Z) in each group.
            closest_depths = np.full(inverse.max() + 1, np.inf)
            np.minimum.at(closest_depths, inverse, Z[valid_indices])

            # Exclude points that are farther away than the closest point.
            exclusionmask[valid_indices] = (
                    Z[valid_indices] == closest_depths[inverse]
            )

        return pixels, exclusionmask

    def extract_window_no_padding(self, image, original_center_x, original_center_y, size=224,plot=True):
        """
        Extract a size x size window centered as close as possible to
        (center_x, center_y), shifting the window if near borders.
        """
        half = size // 2

        h, w = image.shape[:2]

        x_bracket = math.floor(original_center_x/size)
        center_x = x_bracket*224+112
        #y_bracket = math.floor(original_center_y/size)
        y_bracket = int(original_center_y/size)
        center_y = y_bracket*224+112

        if (center_x +112)>=w:
            center_x = w-112
        if (center_y +112)>=h:
            center_y = h-112


        # Shift center so the window fits completely
        #center_x = np.clip(original_center_x, half, w - half)
        #center_y = np.clip(original_center_y, half, h - half)
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


    def transform_gaussians_any(self,key,R,translation):
        means=self.subcenter_dict[key]['means']
        scales =self.subcenter_dict[key]['scales']
        rotations =self.subcenter_dict[key]['rotations']
        harmonics = self.subcenter_dict[key]['harmonics']
        opacities = self.subcenter_dict[key]['opacities']

        rotation = Rotation.from_matrix(R)

        # Transform positions

        self.subcenter_dict[key]['means'] = rotation.apply(self.subcenter_dict[key]['means']) + translation

        # Transform orientations

        gaussian_rot = Rotation.from_quat(rotations)

        new_rot = rotation * gaussian_rot

        self.subcenter_dict[key]['rotations'] = new_rot.as_quat()

        return

    def transform_gaussians_individual(
            self,
            key,
            extrinsics,
            ctr_img_nr,
            cropstarts,
            distance,
            targetpixel,
    ):

        H = W = 448
        S = 224
        x_bracket = math.floor((targetpixel[0]+cropstarts[0])/S)
        y_bracket = int((targetpixel[1]+cropstarts[1])/S)
        R = self.rot_arr[y_bracket,x_bracket]
        # --------------------------------------------------
        # 1. Extract Gaussian parameters
        # --------------------------------------------------

        means=self.subcenter_dict[key]['means'].reshape(H, W, 3)
        scales =self.subcenter_dict[key]['scales']
        rotations =self.subcenter_dict[key]['rotations'].reshape(H, W, 4)
        harmonics = self.subcenter_dict[key]['harmonics']
        opacities = self.subcenter_dict[key]['opacities']

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

        #centerpt = means[target_y, target_x].copy()

        #dist = np.linalg.norm(centerpt)

        #if dist < 1e-12:
        #    raise ValueError("Target Gaussian coincides with the camera origin.")

        #distscale = distance / dist

        #means = means.reshape(-1, 3)  * distscale

        #scales = scales * distscale

        # --------------------------------------
        # 6.5. Experiments regarding multidistance scaling
        # --------------------------------------
        means_inshape = means.reshape(H, W, 3)
        subcenters_dict = self.subcenter_dict[ctr_img_nr[0]]

        vertex_parts = [
            self.centers[ctr_img_nr[0]][None, :],
            self.volumes[ctr_img_nr[0]],
        ]

        for sub_key, subcenter in subcenters_dict.items():
            if sub_key.startswith("subcenter"):
                nr = subcenter["centernumber"]

                vertex_parts.append(
                    self.centers[nr][None, :]
                )
                vertex_parts.append(self.volumes[nr])

        vertices = np.concatenate(vertex_parts, axis=0)

        pixels, mask =self.project_points(ctr_img_nr[1],vertices,cropstarts[0],cropstarts[1])
        pixels_masked, vertices_masked=pixels[mask], vertices[mask]
        depths = np.linalg.norm(vertices_masked-self.poses[ctr_img_nr[1]][0:3, 3:].reshape((3,)), axis=1)

        corrected_pixels = (pixels_masked - np.asarray(cropstarts, dtype=pixels_masked.dtype)) * 2

        pred_vertices = means[
            corrected_pixels[:, 1],
            corrected_pixels[:, 0]
        ]
        pred_depths =np.linalg.norm(pred_vertices, axis=1)
        dscales = depths/pred_depths
        if len(dscales) < 4:
            depth_correctionmap = np.ones((448,448))*dscales[0]
        else:
            depth_correctionmap = self.create_depth_map(corrected_pixels, dscales)
        lowest = np.min(depths)
        biggest = np.max(depths)
        depth_correctionmap = np.clip(depth_correctionmap, lowest, biggest)
        '''if ctr_img_nr[0]==145:
            f, axarr = plt.subplots(2, 1)
            axarr[0].imshow(depth_correctionmap)
            plt.show()'''
        #depth_correctionmap = np.clip(depth_correctionmap,0.8,1.2)
        means_corrected = depth_correctionmap[:, :, np.newaxis] * means_inshape
        means = means_corrected.reshape(-1, 3)
        scales = depth_correctionmap[:, :, np.newaxis]*scales.reshape(448,448,3)
        scales = scales.reshape(448*448,3)
        # --------------------------------------------------
        # 7. Convert back to PyTorch tensors
        # --------------------------------------------------
        self.subcenter_dict[key]['means']=means
        self.subcenter_dict[key]['scales']=scales
        self.subcenter_dict[key]['rotations']=rotations.reshape(-1, 4)
        self.subcenter_dict[key]['harmonics']=harmonics
        self.subcenter_dict[key]['opacities']=opacities

        target_mean = means[target_y * W + target_x]

        pixel00 =  np.array([cropstarts[0] ,cropstarts[1] ,])
        pixel01 =  np.array([cropstarts[0]+224 ,cropstarts[1] ,])
        pixel10 =  np.array([cropstarts[0] ,cropstarts[1]+224 ,])
        pixel11 =  np.array([cropstarts[0]+224 ,cropstarts[1]+224 ,])
        origin00,direction00 = self.pixel_to_world_ray(pixel00, self.K, extrinsics)
        origin01,direction01 = self.pixel_to_world_ray(pixel01,  self.K, extrinsics)
        origin10,direction10 = self.pixel_to_world_ray(pixel10,  self.K, extrinsics)
        origin11,direction11 = self.pixel_to_world_ray(pixel11,  self.K, extrinsics)
        zeroverts=np.array([origin00+direction00,origin01+direction01,origin10+direction10,origin11+direction11])

        origin00p,direction00p = self.pixel_to_world_ray(pixel00, self.K, self.poses[ctr_img_nr[1]])
        origin01p,direction01p = self.pixel_to_world_ray(pixel01,  self.K, self.poses[ctr_img_nr[1]])
        origin10p,direction10p = self.pixel_to_world_ray(pixel10,  self.K, self.poses[ctr_img_nr[1]])
        origin11p,direction11p = self.pixel_to_world_ray(pixel11,  self.K, self.poses[ctr_img_nr[1]])
        pverts=np.array([origin00p+direction00p,origin01p+direction01p,origin10p+direction10p,origin11p+direction11p])


        return target_mean, zeroverts, pverts
        return newgaussians, target_mean, target_means, target_verts

    def create_smooth_depth_map_rbf(
            self, pixel_coords, depths, size=448, neighbors=32
    ):
        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64).reshape(-1)

        # Generate pixel coordinates.
        x, y = np.meshgrid(
            np.arange(size),
            np.arange(size),
            indexing="xy"
        )

        grid_points = np.column_stack((x.ravel(), y.ravel()))

        # Local RBF interpolation.
        interpolator = RBFInterpolator(
            pixel_coords,
            depths,
            kernel="thin_plate_spline",
            smoothing=10.0,
            neighbors=neighbors
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

    def overlap_masks(self):
        maks_dict={}

        for key in self.subcenter_dict.keys():
            mask = np.ones((448,448))
            isedge=((int(self.subcenter_dict[key]['crop_start']['x'])%224)!=0)or(int(self.subcenter_dict[key]['crop_start']['y'])==0)

            if not isedge:
                mask = np.array(mask, dtype=bool).reshape((448 * 448,))
                maks_dict[key] = mask
                continue
            for key2 in self.subcenter_dict.keys():
                if key==key2:
                    continue
                if self.subcenter_dict[key]['imagenr']!=self.subcenter_dict[key2]['imagenr']:
                    continue

                cropstartx1 = int(self.subcenter_dict[key]['crop_start']['x'])
                cropstartx2 = int(self.subcenter_dict[key2]['crop_start']['x'])
                cropstarty1 = int(self.subcenter_dict[key]['crop_start']['y'])
                cropstarty2 = int(self.subcenter_dict[key2]['crop_start']['y'])
                if ((cropstartx2+224)>cropstartx1) and (cropstarty1==cropstarty2)and (cropstartx1+224>=self.img_w):
                    mask[:,:(cropstartx2-cropstartx1+224)*2] =mask[:,:(cropstartx2-cropstartx1+224)*2]*0
                    continue
                if (cropstarty2<224) and (cropstartx1==cropstartx2)and (cropstarty1==0):
                    mask[cropstarty2*2:,:]=mask[cropstarty2*2:,:]*0
                    continue

            mask = np.array(mask, dtype=bool).reshape((448*448,))
            maks_dict[key] = mask
        return maks_dict

    def postprocess(self):
        pairlist=[]

        for key in self.subcenter_dict.keys():
            imgnr1 = int(self.subcenter_dict[key]['imagenr'])
            cropstartx1 = int(self.subcenter_dict[key]['crop_start']['x'])
            for key2 in self.subcenter_dict.keys():
                if (key==key2):
                    continue
                imgnr2 = int(self.subcenter_dict[key2]['imagenr'])
                cropstartx2 = int(self.subcenter_dict[key2]['crop_start']['x'])
                if (imgnr1==imgnr2) and (cropstartx1+224==cropstartx2):
                    pairlist.append([key,key2])
                    break

        for pair in pairlist:
            imgnr = int(self.subcenter_dict[pair[0]]['imagenr'])
            means1 = self.subcenter_dict[pair[0]]['means'].reshape(448,448,3)
            means2 = self.subcenter_dict[pair[1]]['means'].reshape(448,448,3)
            scales1 = self.subcenter_dict[pair[0]]['scales'].reshape(448,448,3)
            scales2 = self.subcenter_dict[pair[1]]['scales'].reshape(448,448,3)
            extrinsics = self.poses[imgnr][0:3, 3:].reshape((3,))
            edge_means1=means1[:,-1,:]
            edge_means2=means1[:,0,:]
            distances1 =np.linalg.norm(edge_means1 - extrinsics, axis=1)
            distances2 =np.linalg.norm(edge_means2 - extrinsics, axis=1)
            distance_ratio1= ((distances2/distances1)/2)+0.5
            distance_ratio2= ((distances1/distances2)/2)+0.5
            depthmap1=np.ones((448,448))
            depthmap2=np.ones((448,448))
            for i in range(448):
                row1 = np.linspace(1, distance_ratio1[i], num=448)
                row2 = np.linspace(distance_ratio2[i], 1, num=448)
                depthmap1[i,:]=row1
                depthmap2[i,:] = row2

            means1 = depthmap1[:, :, np.newaxis] * means1
            self.subcenter_dict[pair[0]]['means'] = means1.reshape(-1, 3)
            scales1 = depthmap1[:, :, np.newaxis]*scales1
            self.subcenter_dict[pair[0]]['scales'] = scales1.reshape(448*448,3)
            means2 = depthmap2[:, :, np.newaxis] * means2
            self.subcenter_dict[pair[1]]['means'] = means2.reshape(-1, 3)
            scales2 = depthmap2[:, :, np.newaxis]*scales2
            self.subcenter_dict[pair[1]]['scales'] = scales2.reshape(448*448,3)

        return
    def return_nextscenemask(self, means):
        poses = []
        path = Path(self.config.pose_path)
        with open(path) as f:
            for line in f:
                vals = np.fromstring(line.strip(), sep=' ')
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = vals.reshape(3, 4)
                poses.append(T)
        pose = np.stack(poses)[self.frame_end,:,:]
        inv_pose = np.linalg.inv(pose)
        posedmeans = means.copy()

        N = len(posedmeans)
        pts_c = (inv_pose @ np.hstack([posedmeans, np.ones((N, 1))]).T).T
        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        #valid = (Z < 0)
        valid = (Z > NEAR_CLIP) & (Z < FAR_CLIP)
        u = (self.K[0, 0] * X / Z + self.K[0, 2]).astype(np.int32)
        v = (self.K[1, 1] * Y / Z + self.K[1, 2]).astype(np.int32)

        #in_img = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        in_img = (u >= 0) & (u < self.img_w)  & (v < self.img_h)

        exclusionmask = np.invert(in_img * valid)
        return exclusionmask

    def submap_gs_stitching_pipeline(self):
        flag=True
        for key in self.subcenter_dict.keys():
            print(key)
            overlapmask = self.overlap_dict[key]
            if flag:
                flag=False
                combined_means = self.subcenter_dict[key]['means'][overlapmask]
                combined_scales = self.subcenter_dict[key]['scales'][overlapmask]
                combined_rotations = self.subcenter_dict[key]['rotations'][overlapmask]
                combined_harmonics = self.subcenter_dict[key]['harmonics'][overlapmask]
                combined_opacities = self.subcenter_dict[key]['opacities'][overlapmask]
            else:
                combined_means = np.concatenate([combined_means, self.subcenter_dict[key]['means'][overlapmask]])
                combined_scales = np.concatenate([combined_scales, self.subcenter_dict[key]['scales'][overlapmask]])
                combined_rotations = np.concatenate([combined_rotations, self.subcenter_dict[key]['rotations'][overlapmask]])
                combined_harmonics = np.concatenate([combined_harmonics, self.subcenter_dict[key]['harmonics'][overlapmask]])
                combined_opacities = np.concatenate([combined_opacities, self.subcenter_dict[key]['opacities'][overlapmask]])
        print('average scale: ',np.average(combined_scales))
        print('max scale: ',np.max(combined_scales))
        largescale_mask = np.max(combined_scales,axis=1)<0.2

        if not self.frame_end>=self.maxlen-2:
            nextscenemask = self.return_nextscenemask(combined_means)*largescale_mask

            combined_means_tm=torch.tensor(combined_means[nextscenemask], dtype=torch.float32, device="cuda")[None]
            combined_scales_tm = torch.tensor(combined_scales[nextscenemask], dtype=torch.float32, device="cuda")[None]
            combined_rotations_tm = torch.tensor(combined_rotations[nextscenemask], dtype=torch.float32, device="cuda")[None]
            combined_harmonics_tm = torch.tensor(combined_harmonics[nextscenemask], dtype=torch.float32, device="cuda")[None]
            combined_opacities_tm = torch.tensor(combined_opacities[nextscenemask], dtype=torch.float32, device="cuda")[None]
            combined_covariances_tm = build_covariance(combined_scales_tm,combined_rotations_tm)
        else:
            combined_means_tm = torch.tensor(combined_means, dtype=torch.float32, device="cuda")[None]
            combined_scales_tm = torch.tensor(combined_scales, dtype=torch.float32, device="cuda")[None]
            combined_rotations_tm = torch.tensor(combined_rotations, dtype=torch.float32, device="cuda")[None]
            combined_harmonics_tm = torch.tensor(combined_harmonics, dtype=torch.float32, device="cuda")[None]
            combined_opacities_tm = torch.tensor(combined_opacities, dtype=torch.float32, device="cuda")[None]
            combined_covariances_tm = build_covariance(combined_scales_tm, combined_rotations_tm)

        combined_gaussians= Gaussians(
            means=combined_means_tm,
            covariances=combined_covariances_tm,
            harmonics=combined_harmonics_tm,
            opacities=combined_opacities_tm,
            scales=combined_scales_tm,
            rotations=combined_rotations_tm)
        end = time.perf_counter()
        print(f"Execution time: {end - self.start:.4f} seconds")
        savepath = Path(self.config.output_dir+'/gs_submaps_r'+str(self.config.vol_radius)+'/gs'+self.submapname)
        export_ply(combined_gaussians.means[0], combined_gaussians.scales[0], combined_gaussians.rotations[0], combined_gaussians.harmonics[0],
                   combined_gaussians.opacities[0], savepath)

        return end - self.start
def load_ply_as_anysplat(ply_path, device="cuda"):
    # Read PLY file
    plydata = PlyData.read(ply_path)
    print('loaded')
    vertex = plydata["vertex"].data
    properties = vertex.dtype.names

    def get_property(name):
        if name not in properties:
            raise ValueError(f"Missing PLY property: {name}")
        return np.asarray(vertex[name], dtype=np.float32)

    # Positions: (N, 3)
    means = np.stack([
        get_property("x"),
        get_property("y"),
        get_property("z")
    ], axis=-1)

    # Scales: PLY stores log(scales), so convert back
    scales = np.stack([
        get_property("scale_0"),
        get_property("scale_1"),
        get_property("scale_2")
    ], axis=-1)
    scales = np.exp(scales)

    # Rotations: PLY stores w, x, y, z.
    # AnySplat internally uses x, y, z, w.
    rotations = np.stack([
        get_property("rot_1"),
        get_property("rot_2"),
        get_property("rot_3"),
        get_property("rot_0")
    ], axis=-1)

    # Normalize quaternions
    rotations /= np.maximum(
        np.linalg.norm(rotations, axis=-1, keepdims=True),
        1e-8
    )

    # Opacity: stored as logits in standard 3DGS PLY files
    opacity_logits = get_property("opacity")
    opacities = 1.0 / (1.0 + np.exp(-opacity_logits))

    # Spherical harmonics
    f_dc = np.stack([
        get_property("f_dc_0"),
        get_property("f_dc_1"),
        get_property("f_dc_2")
    ], axis=-1)

    # AnySplat expects harmonics shaped (N, 3, d_sh).
    # Its own exporter may save only the DC coefficients.
    if all(f"f_rest_{i}" in properties for i in range(3 * 15)):
        f_rest = np.stack([
            get_property(f"f_rest_{i}")
            for i in range(3 * 15)
        ], axis=-1)

        # Standard 3DGS stores higher-order SH coefficients
        # channel-major. Reshape to (N, 3, 15).
        f_rest = f_rest.reshape(-1, 3, 15)

        harmonics = np.concatenate(
            [f_dc[:, :, None], f_rest],
            axis=-1
        )
    else:
        # DC-only PLY: remaining SH coefficients are unavailable.
        harmonics = f_dc[:, :, None]

    # Convert to PyTorch tensors
    means = torch.from_numpy(means).to(device)
    scales = torch.from_numpy(scales).to(device)
    rotations = torch.from_numpy(rotations).to(device)
    opacities = torch.from_numpy(opacities).to(device)
    harmonics = torch.from_numpy(harmonics).to(device)

    # Construct covariance matrices from scales and rotations
    covariances = build_covariance(scales, rotations)

    # Add batch dimension
    gaussians = Gaussians(
        means=means.unsqueeze(0),
        covariances=covariances.unsqueeze(0),
        harmonics=harmonics.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        scales=scales.unsqueeze(0),
        rotations=rotations.unsqueeze(0),
    )

    return gaussians
def stitching_submap_gs(framestarts, frameends,config):
    plyfolder = Path(config.output_dir+'/gs_submaps_r'+str(config.vol_radius))
    plynames=sorted(plyfolder.glob("*.ply"))
    flag = True
    for ply_file in plynames:
        #frameend = frameends[j]
        #if j > 5:
        #    break
        #path = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussiansubmapstitch"+str(config.radius)+"mmultic_yolocrops_egdeallign_noverlap_"+ str(framestarts[j]+1) +".ply"
        ply = PlyData.read(ply_file)

        vetices = ply["vertex"].data
        rotated_verts = vetices
        if flag:
            flag=False
            corrected_verts = rotated_verts
        else:
            corrected_verts = np.concatenate([corrected_verts, rotated_verts])
        print(str(ply_file.stem))

    ply_out = PlyData([
        PlyElement.describe(corrected_verts, "vertex")
    ], text=False)
    ply_out.write(
        config.output_dir+'/gs_global_r'+str(config.vol_radius)+'/global_gaussians.ply')
    return

def pixel_to_world_ray(pixel, K, extrinsics):

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

def precalc_rotations(config):

    csv_path = Path(config.output_dir+'/utils/'+config.output_dir.split('/')[-1]+'_rotations.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    img_h = config.img_h
    img_w = config.img_w
    size = 224
    x_bracket_max = math.floor(img_w / size)
    y_bracket_max = int(img_h / size)
    original_shape = (y_bracket_max + 1, x_bracket_max + 1, size, size, 3, 3)

    if csv_path.exists():
        # Load existing CSV as a NumPy array
        data = np.loadtxt(csv_path, delimiter=",")
        data = data.reshape(original_shape)
    else:
        # Create an empty array
        zeroextrinsics = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        calib=pd.read_csv(config.calib_path, delimiter=' ', header=None, index_col=0)
        p2 = np.array(calib.loc['P2:']).reshape((3, 4))
        intr, r2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(p2)
        data = np.empty((y_bracket_max + 1, x_bracket_max + 1, size, size, 3, 3), dtype=np.float64)

        for x_bracket in range(x_bracket_max+1):
            for y_bracket in range(y_bracket_max+1):
                directions_from = np.empty((size, size, 3), dtype=np.float64)
                directions_to = np.empty((size, size, 3), dtype=np.float64)
                for x_from in range(size):
                    for y_from in range(size):

                        if (x_bracket*224+224)>=img_w:
                            xto = img_w-224+x_from
                        else:
                            xto = x_bracket * size + x_from
                        if (y_bracket*224+224)>=img_h:
                            yto = img_h -224 + y_from
                        else:
                            yto = y_bracket * size + y_from
                        pixel_a = np.array([intr[0, 2] - 112 + x_from,intr[1, 2] - 112 + y_from,])

                        pixel_b = np.array([xto,yto,])
                        origin_from, directions_from[y_from, x_from] = pixel_to_world_ray(pixel_a, intr, zeroextrinsics)

                        origin_to, directions_to[y_from, x_from] = pixel_to_world_ray(pixel_b, intr, zeroextrinsics)
                a = directions_from
                b = directions_to

                a = a / np.linalg.norm(a, axis=-1, keepdims=True)
                b = b / np.linalg.norm(b, axis=-1, keepdims=True)

                cross = np.cross(a, b)
                dot = np.sum(a * b, axis=-1)

                # Skew-symmetric matrices for all pixel rotations
                K = np.zeros((size, size, 3, 3), dtype=np.float64)

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

                R = (I + K + K2 / denominator[..., None, None])
                data[y_bracket,x_bracket,:,:,:] = R
        # Create parent directories if they don't exist
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the array as a CSV file
        np.savetxt(csv_path, data.reshape(-1, original_shape[-1]), delimiter=",")

    return data

def init_model():
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    model.encoder.cfg.voxelize=False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def singlesubmap(meshpath,config, model, framestart, frameend,rot_arr):
    submap = Submap(config, framestart, frameend, model,rot_arr,meshpath)
    submap.init_geometry()
    #submap.init_model()
    submap.init_model_inputs()
    #submap.pipeline_foward(visual_toggle=True)
    submap.pipeline_foward()
    #submap.postprocess()
    time = submap.submap_gs_stitching_pipeline()
    return time

def multisubmap(config):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.output_dir+'/utils', exist_ok=True)
    os.makedirs(config.output_dir+'/renders_r'+str(config.vol_radius), exist_ok=True)
    os.makedirs(config.output_dir+'/gs_submaps_r'+str(config.vol_radius), exist_ok=True)
    os.makedirs(config.output_dir+'/gs_global_r'+str(config.vol_radius), exist_ok=True)

    model = init_model()


    rot_arr = precalc_rotations(config)

    imgfolder = Path(config.image_path)
    imgnames=sorted(imgfolder.glob("*.png"))
    #for i in range(len(imgnames)):

    plyfolder = Path(config.submap_folder)
    plynames=sorted(plyfolder.glob("*.ply"))

    framestarts = []
    frameends = []
    for ply_file in plynames:
        print(ply_file.stem[-6:])
        framestarts.append(int(ply_file.stem[-6:])-1)
        #if len(framestarts)>5:
        #    break
    for i in range(len(framestarts)-1):
        frameends.append(framestarts[i+1])
    frameends.append(int(list(imgnames)[-1].stem[-6:]))
    time = 0
    for i in range(len(framestarts)):
        #if i > 7:
        #    break
        print(list(plynames)[i])
        print(framestarts[i])
        print(frameends[i])# Full path
        #if i == 0 or i ==1:
        #    continue
        time+=singlesubmap(str(list(plynames)[i]), config,model,framestarts[i], frameends[i],rot_arr)
    print('total executiontime: ',time)
    #stitching_submap_gs(framestarts,frameends,config)

    return
def render_splat_ret_img(means,scale,rot,harmonics, opacity,covariance,extrinsics,intrinsics,h=224*2,w=224*2):

    extrinsics_tensor = torch.tensor(np.linalg.inv(extrinsics), dtype=torch.float32, device="cuda")[None]
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device="cuda")[None]
    #bgc = np.array([135, 206, 235])/255 #blue
    #background_color = torch.tensor(bgc, dtype=torch.float32, device="cuda")[None][0]
    #background_color = torch.tensor(np.array([1, 1, 1]), dtype=torch.float32, device="cuda")[None][0]
    background_color = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device="cuda")[None][0]
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
    start=0
    end=100
    #imges = []
    gaussians = load_ply_as_anysplat(config.output_dir+'/gs_global_r'+str(config.vol_radius)+'/global_gaussians.ply')

    for i in range(len(poses)):
        print(i)
        render=render_splat_ret_img(gaussians.means,gaussians.scales,gaussians.rotations,gaussians.harmonics,gaussians.opacities,gaussians.covariances,poses[i],K,config.img_h,config.img_w)
        render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
        #imges.append(render[0])
        cv2.imwrite(config.output_dir + '/rendersb_r' + str(config.vol_radius)+'/gsrender' + str(i).zfill(6) + '.png', render)

    return


def load_config(file_path: str = "config.yaml") -> Config:
    """Load a YAML file into a Config object."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}

    return Config(**data)
def main():
    configpath = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/kittiseq10_config2.yaml"
    config = load_config(configpath)
    multisubmap(config)
    #render_scene(config)
    return


if __name__ == '__main__':
    sys.exit(main())