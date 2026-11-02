import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class PoseOptimizer:

    def __init__(self, reference_image, render_fn, loss_fn,substitute_mesh_verts,extrinsics,intrinsics,initial_translate):
        self.reference_image = reference_image
        self.render_fn = render_fn
        self.loss_fn = loss_fn
        self.substitute_mesh_verts=substitute_mesh_verts
        self.extrinsics=extrinsics
        self.intrinsics=intrinsics
        self.initial_translate =initial_translate


    def pose_to_matrix(self, pose):
        """
        pose = [rx, ry, rz, tx, ty, tz]

        Rotation is represented as an axis-angle vector.
        """

        R = Rotation.from_rotvec(pose[:3]).as_matrix()
        t = pose[3:]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def objective(self, pose):
        T = self.pose_to_matrix(np.array([pose[0],pose[1],pose[2],self.initial_translate[0],self.initial_translate[1],self.initial_translate[2]]))

        rendered = self.render_fn(T,self.substitute_mesh_verts, self.extrinsics,self.intrinsics)

        return self.loss_fn(
            self.reference_image,
            rendered
        )

    def optimize(self, initial_pose):
        result = minimize(
            self.objective,
            initial_pose,
            method="Powell",
            bounds=[[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],
            options={
                "maxiter": 500,
                "disp": True
            }
        )

        #transformation = self.pose_to_matrix(result.x)
        transformation = self.pose_to_matrix(np.array([result.x[0],result.x[1],result.x[2],self.initial_translate[0],self.initial_translate[1],self.initial_translate[2]]))

        return transformation, result