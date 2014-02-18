"""
This module contains functions to create seeding and target rois for 
connectivity analyses.
"""
from __future__ import division
import numpy as np
from ..mr_simul.partial_volume import compute_corner_positions


def spherical_shell_mask(voxel_size, image_size, center, radius,
                         shell_thickness):
    """
    Returns a mask of a spherical shell, with selected (outer) radius and 
    thickness.

    Parameters
    ----------
    voxel_size : double
        voxel size in mm (only isotropic voxels are supported).
    image_size : double
        image size in mm (we assume image has cubic shape).
    center : array-like, shape (3, )
    radius : double
        spherical shell outer radius in mm.
    shell_thickness : double
        spherical shell thickness in mm.
    """
    dim_x = dim_y = dim_z = int(image_size / voxel_size)

    corner_positions = compute_corner_positions(voxel_size, image_size)

    center_to_corners = corner_positions - center
    dst_to_corners = (point_to_corners ** 2).sum(-1)

    out_radius = radius
    in_radius = radius - shell_thickness

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


def _merge_patches(fibers):
    """
    From a set of fibers, identify the end points that need to be merged to
    define the connectivity rois.

    Parameters
    ----------
    fibers : sequence
        A list of phantomas.geometry.models.Fiber objects.
    
    Returns
    -------
    label_indices : sequence
        The sequence of labels, such as the endpoints of fiber i are
        associated to label_indices[2*i] and label_indices[2*i + 1], 
        respectively.
    connectivity_matrix : array-like, shape (nb_regions, nb_regions)
        The binary connectivity matrix. Warning: labels start from 1,
        therefore connectivity_matrix[i, j] indicates whether label (i + 1)
        is connected to label (j + 1).
    """
    nb_endpoints = 2 * len(fibers)
    endpoints = np.zeros((nb_endpoints, 3))
    radii = np.zeros(nb_endpoints)
    for i, fiber in enumerate(fibers):
        fiber_points = fiber.get_points()
        fiber_radius = fiber.get_radius()
        endpoints[2 * i] = fiber_points[0]
        endpoints[2 * i + 1] = fiber_points[-1]
        radii[2 * i] = fiber_radius
        radii[2 * i + 1] = fiber_radius
    endpoints_to_center = np.sqrt(np.sum(endpoints ** 2, 1))
    thetas = np.arctan2(endpoints_to_center, radii)
    endpoints_norm = np.sqrt(np.sum(endpoints ** 2, 1))
    angles  = np.outer(endpoints[:, 0] / endpoints_norm, 
                       endpoints[:, 0] / endpoints_norm)
    angles += np.outer(endpoints[:, 1] / endpoints_norm, 
                       endpoints[:, 1] / endpoints_norm)
    angles += np.outer(endpoints[:, 2] / endpoints_norm, 
                       endpoints[:, 2] / endpoints_norm)
    angles -= thetas[:, np.newaxis]
    angles -= thetas[np.newaxis, :]
    
    labels = [[i] for i in range(1, nb_endpoints + 1)]
    nb_merged = 0 # The number of merged patches so far.
    for i in range(nb_endpoints):
        index_label = i - nb_merged
        merged = False
        for j in range(index_label):
            for id_endpoint in labels[j]:
                if angles[i, id_endpoint] <= 0:
                    # merge patch i to label j
                    nb_merged += 1
                    if not merged:
                        # merge patch i to label j
                        labels[j].append(i)
                        labels.remove(labels[index_label])
                        merged = True
                        merged_to = j
                    else:
                        labels[merged_to].extend(labels[j])
                        labels.remove(labels[j])
    nb_labels = len(labels)
    connectivity_matrix = np.zeros((nb_labels, nb_labels), dtype=np.bool)
    



def target_rois(fibers, voxel_size, image_size):
    """
    From a set of fibers, returns a list of target regions to use for
    connectivity analysis, as well as the corresponding connectivity matrix.

    Parameters
    ----------
    fibers : sequence
        A list of phantomas.geometry.models.Fiber objects.

    Returns
    -------
    labels : array-like, shape (dim_x, dim_y, dim_z)
        The labelmap.
    connectivity_matrix : array-like, shape (nb_regions, nb_regions)
        The binary connectivity matrix. Warning: labels start from 1,
        therefore connectivity_matrix[i, j] indicates whether label (i + 1)
        is connected to label (j + 1).
    """
    # We first identify the intersecting patches
    
    return

