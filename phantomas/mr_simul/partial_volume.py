"""
This module contains functions to compute partial volume effect, for fibers and
other geometrical primitives.
"""
import numpy as np


def compute_affine_matrix(voxel_size, image_size):
    """
    Creates an affine matrix to compute real-world coordinates out of pixel
    coordinates. The convention is to return the coordinates of the center of the
    voxel.

    Parameters
    ----------
    voxel_size : double
        The voxel size, in mm. NB: We assume isotropic voxel size.
    image_size : double
        The image size, in mm. NB: We assume the image has cubic dimensions.

    Returns
    -------
    affine : array-lie, shape (4, 4)
        The Nifti-like affine matrix.
    """
    affine = np.eye(4)
    affine[:3, :3] = voxel_size * np.eye(3)
    actual_image_size = int(image_size / voxel_size) * voxel_size
    affine[:3, 3] = -0.5 * actual_image_size + 0.5 * voxel_size
    return affine


def compute_corner_positions(voxel_size, image_size):
    """
    Computes the positions of voxel corners.
    
    Parameters
    ----------
    voxel_size : double
        The voxel size, in mm.
    image_size : array-like (3,)
        The image size, in mm.

    Returns
    -------
    corner_positions : array-like, shape ``(dim_x+1, dim_y+1, dim_z+1, 3)``
    """
    dim_x = dim_y = dim_z = int(image_size / voxel_size)
    affine = compute_affine_matrix(voxel_size, image_size)

    indices = np.mgrid[0:dim_x + 1, 0:dim_y + 1, 0:dim_z + 1]
    indices = np.rollaxis(indices, 0, 4)

    corner_positions = np.dot(indices, affine[:3, :3].T) \
                     + affine[np.newaxis, np.newaxis, np.newaxis, :3, 3] \
                     - 0.5 * voxel_size
    return corner_positions


def compute_fiber_masks(fibers, voxel_size, image_size):
    '''
    Creates masks of the fibers. Voxels completely filled by the bundle are 
    set to 2. Voxels which are intersected but not filled by the bundle are 
    set to 1. Background is set to 0.
    
    Parameters
    ----------
    fibers : sequence, length (F, )
        A sequence of Fiber instance (see :class:`phantomas.geometry.models.Fiber`).
    voxel_size : int
        the voxel size in mm (voxels are isotropic).
    image_size : int
        the image size in mm (image is cubic).
    '''
    dim_x = dim_y = dim_z = int(image_size / voxel_size)

    corner_positions = compute_corner_positions(voxel_size, image_size)
    corner_indices = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    nb_fibers = len(fibers)
    mask = np.zeros((dim_x, dim_y, dim_z, nb_fibers), dtype=np.uint8)

    for i, fiber in enumerate(fibers):
        radius = fiber.get_radius()
        nb_points = fiber.get_nb_points()
        for n in range(nb_points):
            point = fiber.get_points()[n]

            point_to_corners = corner_positions - point
            dst_to_corners = (point_to_corners ** 2).sum(-1)

            intersected = np.zeros((dim_x, dim_y, dim_z), dtype=np.bool)
            filled = np.ones((dim_x, dim_y, dim_z), dtype=np.bool)
            for x, y, z in corner_indices:
                mask_corner = dst_to_corners[x:dim_x+x, y:dim_y+y, z:dim_z+z] \
                              < radius ** 2
                np.logical_or(intersected, mask_corner, out=intersected)
                np.logical_and(filled, mask_corner, out=filled)

            mask_intersected = np.logical_or(mask[..., i] == 1, intersected)
            mask_filled = np.logical_or(mask[..., i] == 2, filled)
            mask[mask_intersected, i] = 1
            mask[mask_filled, i] = 2
    return mask.reshape((dim_x, dim_y, dim_z, nb_fibers))


def compute_spherical_region_masks(centers, radii, voxel_size, image_size):
    '''
    Creates a mask of a list of spherical regions. 
    Voxels completely filled by the region are set to 2.
    Voxels which are intersected by the region are set to 1.
    Background is set to 0.
    
    Parameters
    ----------
    centers : sequence of ndarray (3, )
        Centers of the spherical regions.
    radii : sequence 
        Radii of the spherical regions.
    voxel_size : int
        the voxel size in mm (voxels are isotropic).
    image_size : int
        the image size in mm (image is cubic).
    '''
    zoom = image_size / 2
    dim_x = dim_y = dim_z = int(image_size / voxel_size)
    dim_image = dim_x * dim_y * dim_z
    affine = compute_affine_matrix(voxel_size, image_size)

    indices = np.mgrid[0:dim_x, 0:dim_y, 0:dim_z]
    center_positions = np.dot(affine[:3, :3], indices.reshape(3, dim_image)).T \
                     + affine[:3, 3]

    corners = np.asarray([[1, 1, 1], [1, 1, -1], 
                          [1, -1, 1], [1, -1, -1], 
                          [-1, 1, 1], [-1, 1, -1], 
                          [-1, -1, 1], [-1, -1, -1]])

    nb_regions = len(radii)
    mask = np.zeros((dim_image, nb_regions), dtype=np.uint8)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        center_to_centers = center_positions - center
        dst_to_centers = (center_to_centers ** 2).sum(1)
    
        center_to_corners = center_to_centers.reshape((dim_image, 3, 1)) \
                         + 0.5 * voxel_size * corners.T
        dst_to_corners = (center_to_corners ** 2).sum(1)
        mask_intersected = np.logical_or(mask[..., i], np.any(dst_to_corners < radius ** 2, axis=1))
        mask_filled = np.logical_or(mask[..., i], np.all(dst_to_corners < radius ** 2, axis=1))
    
        mask[mask_intersected, i] = 1
        mask[mask_filled, i] = 2

    return mask.reshape((dim_x, dim_y, dim_z, nb_regions))


def spherical_regions_volume_fraction(centers, radii, voxel_center, voxel_size, resolution=10):
    """
    Computes the volume fraction of a set of spherical regions.
    
    Parameters
    ----------
    centers : sequence of ndarray (3, ), centers of the spherical regions
    radii : sequence of radius of the spherical regions
    voxel_center : ndarray, shape (3,)
    voxel_size : the voxel size in millimeter
    resolution : int
        the number of subdivision in each dimension.
    """
    subvoxel_size = 1.0 * voxel_size / resolution
    dim_grid = resolution * resolution * resolution
    indices = np.mgrid[0:resolution, 0:resolution, 0:resolution]
    center_positions = voxel_size / resolution * indices.reshape(3, dim_grid).T \
                     - 0.5 * voxel_size + 0.5 * subvoxel_size \
                     + voxel_center

    nb_regions = len(radii)
    volume_fraction = np.zeros(dim_grid, dtype=np.uint8)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        center_to_center = center_positions - center
        dst_to_center = (center_to_center ** 2).sum(1)
        volume_fraction = np.logical_or(volume_fraction, dst_to_center < radius ** 2)
    return volume_fraction.sum() / dim_grid
 

def fibers_volume_fraction(fibers, intersect_codes, 
                           voxel_center, voxel_size, resolution=10):
    '''
    Given a set of fibers and a voxel, returns the volume fraction of each fiber,
    the volume fraction of free water and the volume fraction of gray matter.
    The voxel is subdivided into a grid.

    Parameters
    ----------
    fibers : a sequence of Fiber instances
    intersect_codes : a sequence of int (see Notes)
    voxel_center : ndarray, shape (3,)
    voxel_size : the voxel size in millimeter
    resolution : int
        the number of subdivision in each dimension.

    Notes
    -----
    intersect_codes[i] informs whether fiber i does not cross the voxel (0),
    simply intersects the voxel (1), or intersects the whole voxel (2).
    '''
    nb_fibers = len(fibers)
    subvoxel_size = 1.0 * voxel_size / resolution
    dim_grid = resolution * resolution * resolution
    indices = np.mgrid[0:resolution, 0:resolution, 0:resolution]
    center_positions = voxel_size / resolution * indices.reshape(3, dim_grid).T \
                     - 0.5 * voxel_size + 0.5 * subvoxel_size \
                     + voxel_center
    fibers_volume_fraction = np.zeros((dim_grid, nb_fibers))
    for i, fiber in enumerate(fibers):
        if intersect_codes[i] == 0:
            pass
        elif intersect_codes[i] == 2:
            fibers_volume_fraction[..., i] = 0.
        else:
            radius = fiber.get_radius()
            nb_points = fiber.get_nb_points()
            for n in range(nb_points):
                point = fiber.get_points()[n]
                point_to_centers = center_positions - point
                dst_to_centers = (point_to_centers ** 2).sum(1)
                fibers_volume_fraction[dst_to_centers < radius ** 2, i] = 1.0
    sum_volume_fraction = fibers_volume_fraction.sum(-1)
    multiple_fibers = sum_volume_fraction > 1
    if np.count_nonzero(multiple_fibers) > 0:
        fibers_volume_fraction[multiple_fibers] /= sum_volume_fraction[multiple_fibers][:, None]
    fibers_volume_fraction = np.sum(fibers_volume_fraction, axis=0) / resolution**3
    return fibers_volume_fraction


def compute_volume_fractions(phantom_center, phantom_radius, gm_mask,
                             fibers, fiber_masks,
                             region_centers, region_radii, region_masks,
                             voxel_size, image_size):
    """
    Computes the volume fraction of background and of each tissue type, for a
    given resolution.

    Parameters
    ----------
    phantom_center : array-like, shape (3, )
    phantom_radius : double
    gm_mask : array-like, shape (dim_x, dim_y, dim_z)
    fibers : sequence of ``phantomas.geometry.models.Fiber``
    fiber_masks : array-like, shape (dim_x, dim_y, dim_z, nb_fibers)
    region_centers : sequence of array-like
    ragion_radii : sequence
    region_masks : array-like, shape (dim_x, dim_y, dim_z, nb_regions)
    voxel_size : double
    image_size : double

    Returns
    -------
    background_volume_fraction : array-like, shape (dim_x, dim_y, dim_z)
    gm_volume_fraction : array-like, shape (dim_x, dim_y, dim_z)
    wm_volume_fraction : array-like, shape (dim_x, dim_y, dim_z)
    csf_volume_fraction : array-like, shape (dim_x, dim_y, dim_z)
    """
    affine = compute_affine_matrix(voxel_size, image_size)

    # Gray Matter volume fraction
    gm_volume_fraction = np.zeros(gm_mask.shape)
    gm_volume_fraction[gm_mask == 2] = 1.0
    gm_partial_volume = gm_mask == 1
    gm_indices = np.nonzero(gm_partial_volume)
    for i, j, k in zip(gm_indices[0], gm_indices[1], gm_indices[2]):
        voxel_center = np.dot(affine, np.asarray([i, j, k, 1.0]))[:3]
        gm_volume_fraction[i, j, k] = \
              spherical_regions_volume_fraction([np.asarray([0., 0., 0.])], 
                                                [phantom_radius], 
                                                voxel_center, voxel_size)
    
    # Background volume fraction
    background_volume_fraction = 1.0 - gm_volume_fraction
    
    # White Matter volume fraction
    wm_volume_fraction = np.zeros(fiber_masks.shape[:3])
    wm_partial_volume = np.any(fiber_masks == 1, axis=-1)
    nb_voxels_partial_volume = np.count_nonzero(wm_partial_volume)
    wm_indices = np.nonzero(wm_partial_volume)
    for i, j, k in zip(wm_indices[0], wm_indices[1], wm_indices[2]):
        voxel_center = np.dot(affine, np.asarray([i, j, k, 1.0]))[:3]
        intersect_codes = fiber_masks[i, j, k]
        wm_volume_fraction[i, j, k] = \
            np.sum(fibers_volume_fraction(fibers, intersect_codes, 
                                          voxel_center, voxel_size))
    wm_volume_fraction[np.any(fiber_masks == 2, axis=-1)] = 1.0
    # mask out anything outside of the phantom
    np.clip(wm_volume_fraction, 0., gm_volume_fraction, wm_volume_fraction)
    # Make sure GM + WM <= 1.
    np.clip(gm_volume_fraction, 0 , 1 - wm_volume_fraction, gm_volume_fraction)
    
    # CSF volume fraction
    csf_volume_fraction = np.zeros(wm_volume_fraction.shape)
    csf_volume_fraction[np.any(region_masks == 2, axis=-1)] = 1.0
    csf_partial_volume = (region_masks == 1)
    csf_indices = np.nonzero(csf_partial_volume)
    for i, j, k in zip(csf_indices[0], csf_indices[1], csf_indices[2]):
        voxel_center = np.dot(affine, np.asarray([i, j, k, 1.0]))[:3]
        csf_volume_fraction[i, j, k] = \
              spherical_regions_volume_fraction(region_centers, region_radii, 
                                                voxel_center, voxel_size)

    # Make sure everything sums to one
    np.clip(wm_volume_fraction, 0., 1.0 - csf_volume_fraction, 
            wm_volume_fraction)
    gm_volume_fraction = 1.0 - background_volume_fraction \
                             - wm_volume_fraction \
                             - csf_volume_fraction

    return background_volume_fraction, gm_volume_fraction, \
           wm_volume_fraction, csf_volume_fraction





    # Gray Matter volume fraction
    gm_volume_fraction = np.zeros(gm_mask.shape)
    gm_volume_fraction[gm_mask == 2] = 1.0
    gm_partial_volume = gm_mask == 1
    gm_indices = np.nonzero(gm_partial_volume)
    for n in range(np.count_nonzero(gm_partial_volume)):
        i = gm_indices[0][n]
        j = gm_indices[1][n]
        k = gm_indices[2][n]
        voxel_center = np.dot(affine, np.asarray([i, j, k, 1.0]))[:3]
        gm_volume_fraction[i, j, k] = \
              spherical_regions_volume_fraction([np.asarray([0., 0., 0.])], 
                                                [phantom_radius], 
                                                voxel_center, args.struct_res)
    
    # Background volume fraction
    background_volume_fraction = 1.0 - gm_volume_fraction
    
    # White Matter volume fraction
    wm_volume_fraction = np.zeros(fiber_masks.shape[:3])
    wm_partial_volume = np.any(fiber_masks == 1, axis=-1)
    nb_voxels_partial_volume = np.count_nonzero(wm_partial_volume)
    wm_indices = np.nonzero(wm_partial_volume)
    for n in range(np.count_nonzero(wm_partial_volume)):
        i, j, k = wm_indices[0][n], wm_indices[1][n], wm_indices[2][n]
        voxel_center = np.dot(affine, np.asarray([i, j, k, 1.0]))[:3]
        intersect_codes = fiber_masks[i, j, k]
        wm_volume_fraction[i, j, k] = \
            np.sum(fibers_volume_fraction(fibers, intersect_codes, 
                                          voxel_center, args.struct_res))
    wm_volume_fraction[np.any(fiber_masks == 2, axis=-1)] = 1.0
    # mask out anything outside of hte phantom
    np.clip(wm_volume_fraction, 0., gm_volume_fraction, wm_volume_fraction)
    # Make sure GM + WM <= 1.
    np.clip(gm_volume_fraction, 0 , 1 - wm_volume_fraction, gm_volume_fraction)
    
    # CSF volume fraction
    csf_volume_fraction = np.zeros(wm_volume_fraction.shape)
    csf_volume_fraction[np.any(region_masks == 2, axis=-1)] = 1.0
    csf_partial_volume = region_masks == 1
    csf_indices = np.nonzero(csf_partial_volume)
    for n in range(np.count_nonzero(csf_partial_volume)):
        i, j, k = csf_indices[0][n], csf_indices[1][n], csf_indices[2][n]
        voxel_center = np.dot(affine, np.asarray([i, j, k, 1.0]))[:3]
        csf_volume_fraction[i, j, k] = \
              spherical_regions_volume_fraction(region_centers, region_radii, 
                                                voxel_center, args.struct_res)
    
    
