import torch
import numpy as np
from PIL import Image

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams
from argparse import ArgumentParser

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

ply_path = "/home/luis/Desktop/PythonProjects/2026-schmid-luis-mt/minisplatting2/output/gaussiansubmapstitchseq0ex.ply"

width = 1920
height = 1080

# Camera intrinsics
fx = 1000.0
fy = 1000.0
cx = width / 2
cy = height / 2

# Camera extrinsics:
# world -> camera transformation
R = np.eye(3)
t = np.array([0.0, 0.0, 0.0])


# -------------------------------------------------------
# Camera class compatible with gaussian_renderer.render()
# -------------------------------------------------------

class SimpleCamera:
    def __init__(self, R, t, fx, fy, cx, cy, width, height):
        self.image_width = width
        self.image_height = height

        self.FoVx = 2 * np.arctan(width / (2 * fx))
        self.FoVy = 2 * np.arctan(height / (2 * fy))

        self.znear = 0.01
        self.zfar = 1000.0

        # Camera center in world coordinates
        camera_center = -R.T @ t

        # world -> camera
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3, 3] = t

        # Gaussian splatting repo uses transposed matrices
        self.world_view_transform = torch.tensor(
            W2C, dtype=torch.float32, device="cuda"
        ).transpose(0, 1)

        # Projection matrix
        tan_half_fovx = np.tan(self.FoVx / 2)
        tan_half_fovy = np.tan(self.FoVy / 2)

        top = tan_half_fovy * self.znear
        bottom = -top
        right = tan_half_fovx * self.znear
        left = -right

        P = torch.zeros((4, 4), dtype=torch.float32)

        P[0, 0] = 2 * self.znear / (right - left)
        P[1, 1] = 2 * self.znear / (top - bottom)

        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)

        P[2, 2] = self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (
            self.zfar - self.znear
        )
        P[3, 2] = 1.0

        self.projection_matrix = P.cuda().transpose(0, 1)

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

        self.camera_center = torch.tensor(
            camera_center,
            dtype=torch.float32,
            device="cuda"
        )


# -------------------------------------------------------
# Load Gaussians
# -------------------------------------------------------

gaussians = GaussianModel(3)
gaussians.load_ply(ply_path)

# -------------------------------------------------------
# Camera
# -------------------------------------------------------

camera = SimpleCamera(
    R,
    t,
    fx,
    fy,
    cx,
    cy,
    width,
    height
)

# -------------------------------------------------------
# Pipeline parameters
# -------------------------------------------------------

parser = ArgumentParser()
pipeline = PipelineParams(parser)
pipe = parser.parse_args([])

# -------------------------------------------------------
# Render
# -------------------------------------------------------

background = torch.tensor(
    [0.0, 0.0, 0.0],
    dtype=torch.float32,
    device="cuda"
)

with torch.no_grad():
    result = render(
        camera,
        gaussians,
        pipe,
        background
    )

image = result["render"]

# C,H,W -> H,W,C
image = (
    image.clamp(0, 1)
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

image = (image * 255).astype(np.uint8)

#Image.fromarray(image).save("render.png")