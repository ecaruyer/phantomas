"""
This module contains functions to create seeding and target rois for 
connectivity analyses.
"""
from __future__ import division
import numpy as np
from ..mr_simul.partial_volume import compute_corner_positions



def cortical_shell_mask(voxel_size, image_size, phantom_center, phantom_radius,
                        shell_thickness):
    """
    Returns a mask of a cortical shell.
    """
    dim_x = dim_y = dim_z = int(image_size / voxel_size)

    corner_positions = compute_corner_positions(voxel_size, image_size)

    center_to_corners = corner_positions - phantom_center
    dst_to_corners = (point_to_corners ** 2).sum(-1)

    out_radius = phantom_radius
    in_radius = phantom_radius - shell_thickness

    corner_indices = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    mask_shell = np.zeros((dim_x, dim_y, dim_z), dtype=np.bool)

    for x, y, z in corner_indices:
        mask_corner_in_out = dst_to_corners[x:dim_x+x, y:dim_y+y, z:dim_z+z] \
                              <= out_radius ** 2
        mask_corner_out_in = dst_to_corners[x:dim_x+x, y:dim_y+y, z:dim_z+z] \
                              >= in_radius ** 2
        mask_corner = np.logical_and(mask_corner_in_out, mask_corner_out_in)

        np.logical_or(mask_shell, mask_corner, out=mask_shell)

    return mask_shell


