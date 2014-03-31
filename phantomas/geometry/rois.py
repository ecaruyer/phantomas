"""
This module contains functions to create seeding and target rois for 
connectivity analyses.
"""
from __future__ import division
import numpy as np
from ..mr_simul.partial_volume import (compute_corner_positions, 
                                       compute_affine_matrix)


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
    dst_to_corners = (center_to_corners ** 2).sum(-1)

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


def _merge_patches(endpoints, radii):
    """
    From a set of fibers, identify the end points that need to be merged to
    define the connectivity rois.

    Parameters
    ----------
    endpoints : array-like, shape (nb_endpoints, 3)
    radii : array-like, shape (nb_endpoints, )
        
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
    nb_endpoints = endpoints.shape[0]
    endpoints_norm = np.sqrt(np.sum(endpoints ** 2, 1))
    thetas = np.arctan2(radii, endpoints_norm)

    angles  = np.outer(endpoints[:, 0] / endpoints_norm, 
                       endpoints[:, 0] / endpoints_norm)
    angles += np.outer(endpoints[:, 1] / endpoints_norm, 
                       endpoints[:, 1] / endpoints_norm)
    angles += np.outer(endpoints[:, 2] / endpoints_norm, 
                       endpoints[:, 2] / endpoints_norm)
    np.clip(angles, -1, 1, out=angles)
    np.arccos(angles, out=angles)

    angles -= thetas[:, np.newaxis]
    angles -= thetas[np.newaxis, :]

    label_indices = []
    for i in range(nb_endpoints):
        merged = False
        for label_index, label_endpoints in enumerate(label_indices):
            for j in label_endpoints:
                if angles[i, j] <= 0:
                    label_indices[label_index].append(i)
                    merged = True
                    break
        if not merged:
            label_indices.append([i])
    nb_labels = len(label_indices)

    connectivity_matrix = np.zeros((nb_labels, nb_labels), dtype=np.bool)
    for i in range(nb_labels):
        patches_i = label_indices[i]
        for j in range(i):
            patches_j = label_indices[j]
            for k in patches_i:
                for l in patches_j:
                    if (l % 2 == 0) and (k - l == 1) \
                    or (k % 2 == 0) and (l - k == 1):
                        connectivity_matrix[i, j] = 1
                        connectivity_matrix[j, i] = 1
    
    return label_indices, connectivity_matrix


def target_rois(fibers, voxel_size, image_size):
    """
    From a set of fibers, returns a list of target regions to use for
    connectivity analysis, as well as the corresponding connectivity matrix.

    Parameters
    ----------
    fibers : sequence
        A list of phantomas.geometry.models.Fiber objects.
    voxel_size : double
        voxel size in mm (only isotropic voxels are supported).
    image_size : double
        image size in mm (we assume image has cubic shape).

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

    label_indices, connectivity_matrix = _merge_patches(endpoints, radii)

    center = np.array([0, 0, 0])
    radius = np.max(np.sqrt(np.sum(endpoints**2, -1)))
    shell_thickness = voxel_size
    
    mask_shell = spherical_shell_mask(voxel_size, image_size, center, radius,
                                      shell_thickness)
    indices_shell = np.nonzero(mask_shell)

    dim_x = dim_y = dim_z = int(image_size / voxel_size)
    dim_image = dim_x * dim_y * dim_z
    affine = compute_affine_matrix(voxel_size, image_size)

    indices = np.mgrid[0:dim_x, 0:dim_y, 0:dim_z]
    indices = np.rollaxis(indices, 0, 4)
    center_positions = np.dot(indices, affine[:3, :3].T) \
                     + affine[:3, 3]
    shell_voxels = center_positions[mask_shell]

    voxels_to_endpoints = (shell_voxels[:, 0, np.newaxis] \
                         - endpoints[np.newaxis, :, 0])**2
    voxels_to_endpoints += (shell_voxels[:, 1, np.newaxis] \
                         - endpoints[np.newaxis, :, 1])**2
    voxels_to_endpoints += (shell_voxels[:, 2, np.newaxis] \
                         - endpoints[np.newaxis, :, 2])**2
    np.sqrt(voxels_to_endpoints, out=voxels_to_endpoints)
    voxels_to_endpoints -= radii[np.newaxis, :]
 
    endpoint_indices = np.argmin(voxels_to_endpoints, axis=1)

    labels_volume = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    indices_shell = np.asarray(indices_shell)
    for label_id, label_endpoints in enumerate(label_indices):
        for label_endpoint in label_endpoints:
            for point_id in np.nonzero(endpoint_indices == label_endpoint)[0]:
                i, j, k = indices_shell[:, point_id]
                labels_volume[i, j, k] = label_id + 1
            
    return labels_volume, connectivity_matrix
