import math

import numpy as np
import torch


def setup_coordinate_grid(grid_size=128, field_size=0.0282):
    """Create a regular two-dimensional coordinate grid."""
    x = torch.linspace(-field_size / 2, field_size / 2, grid_size)
    y = torch.linspace(-field_size / 2, field_size / 2, grid_size)
    y_coord, x_coord = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([y_coord, x_coord], dim=-1)


def build_coordinate_train(
    L,
    angle,
    radius,
    ray_angles,
    time_point,
    speed_of_sound,
    delay_sample,
    dt,
    num_rays,
):
    """Build coordinates for rays offset from a circular detector array."""
    angle_rad = np.deg2rad(angle)
    ray_angles = ray_angles.reshape(num_rays, 1)

    rotated_theta = ray_angles + angle_rad + math.pi
    ray_origin = (
        np.linspace(0, speed_of_sound * time_point * L, L)
        + delay_sample * speed_of_sound * time_point
        + dt * speed_of_sound
    ).reshape(1, L)

    x_coordinate = np.cos(rotated_theta) @ ray_origin
    x_coordinate += radius * np.cos(angle_rad)
    y_coordinate = np.sin(rotated_theta) @ ray_origin
    y_coordinate += radius * np.sin(angle_rad)

    coordinates = np.stack([x_coordinate.T, y_coordinate.T], axis=-1)
    return coordinates.reshape(L, num_rays, 2)
