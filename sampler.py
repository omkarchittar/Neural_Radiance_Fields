import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        
        # This method takes a batch of rays, computes z values to represent points along each ray, and then generates 3D points at those depths. 
        # The resulting ray_bundle is updated to include the sampled points and corresponding lengths. 
        # This procedure is commonly used in ray tracing or rendering to sample points along rays to determine how they interact with a 3D scene or geometry.

        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = "cuda")

        # TODO (1.4): Sample points from z values
        N = ray_bundle.origins.shape[0] # ray_bundle.origins.shape = [N, 3] 
        D = z_vals.shape[0] # zvals.shape = [n_pts_per_ray]
        origins = ray_bundle.origins.unsqueeze(1).repeat(1, D, 1) # repeats the origins along a new dimension, resulting in a shape of (N, 1, 3)
        directions = ray_bundle.directions.unsqueeze(1).repeat(1, D, 1) # does the same for ray directions, resulting in a shape of (N, 1, 3)
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1) # adds a new dimension at the beginning, creating a shape of (1, self.n_pts_per_ray)
        # The .unsqueeze(-1) operation adds an additional dimension, resulting in a shape of (1, self.n_pts_per_ray, 1). This enables broadcasting to generate points along each ray.
        sample_points = origins + z_vals*directions

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}