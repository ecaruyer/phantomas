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

    out_radius = phantom_radius
    in_radius = phantom_radius - shell_thickness

    corner_indices = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    inside_outer_boundary = np.zeros((dim_x, dim_y, dim_z), dtype=np.bool)
    outside_inner_boundary = np.zeros((dim_x, dim_y, dim_z), dtype=np.bool)

    
    

