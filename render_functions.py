import argparse
import os
import sys
import datetime
import time
import math
import json
import torch

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import pytorch3d
import torch

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import OpenGLPerspectiveCameras
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def render_points(filename, points, image_size=256, color=[0.7, 0.7, 1], device=None):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    points_renderer = get_points_renderer(image_size=256,radius=0.01)

    textures = torch.ones(points.size()).to(device)*0.5   # (1, N_v, 3)
    rgb = textures * torch.tensor(color).to(device)  # (1, N_v, 3)

    point_cloud = pytorch3d.structures.pointclouds.Pointclouds(
        points=points, features=rgb
    )

    R, T = look_at_view_transform(10.0, 10.0, 96)


    # Prepare the camera:
    cameras = OpenGLPerspectiveCameras(
        R=R,T=T, device=device
    )

    rend = points_renderer(point_cloud.extend(2), cameras=cameras)

    rend = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    plt.imsave(filename, rend)

    # The .cpu moves the tensor to GPU (if needed).
    return rend