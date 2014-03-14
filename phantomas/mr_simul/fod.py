"""
This module contains functions for the computation of the ground truth fiber
orientation distribution (FOD) from a collection of fibers.
"""
import numpy as np
from scipy.special import legendre
from .fast_volume_fraction import *
import os
from ..utils import shm


def compute_directions(fibers, tangents, fiber_radii, voxel_center, voxel_size, 
                       resolution):
    """Given a set of fibers, and a voxel size, compute the set of 
    orientations on a subgrid with associated weights.

    Parameters
    ----------
    fibers : sequence, length (M, )
        A list of arrays.
    tangents : sequence, length (M, )
        A list of arrays.
    fiber_radii : array-like, shape (M, )
        The radius of each fiber bundle.
    voxel_center : array-like shape (3, )
        The center of the voxel where to compute the directions.
    voxel_size : double
        The voxel size in mm.
    resolution : int
        The resolution of the grid subdivision.

    Returns
    -------
    fod_samples : array-like, shape (P, 3)
        A sequence of diffusion directions, corresponding to local directions
        of fibers on the grid subdivision.
    fod_weights : array-like, shape (P, )
        A sequence of weights, corresponding to the volume fraction (relative 
        to the voxel) of each diffusion direction.

    """
    subvoxel_size = 1.0 * voxel_size / resolution
    dim_grid = resolution * resolution * resolution
    nb_fibers = len(fibers)
    indices = np.mgrid[0:resolution, 0:resolution, 0:resolution]
    center_positions = subvoxel_size * indices.reshape(3, dim_grid).T \
                     - 0.5 * voxel_size + 0.5 * subvoxel_size \
                     + voxel_center
    center_positions = np.ascontiguousarray(center_positions)
    total_nb_fibers = np.zeros(dim_grid)
    compartments = np.zeros((nb_fibers, dim_grid), dtype=np.bool)
    directions = np.zeros((nb_fibers, dim_grid, 3), dtype=np.double)
    for i, fiber, tangent, fiber_radius in zip(range(nb_fibers), fibers, 
                                               tangents, fiber_radii): 
        fiber_indices = in_fiber(center_positions,
                           fiber.copy("C"), 
                           tangent.copy("C"), 
                           fiber_radius)
        compartments[i] = fiber_indices > -1
        directions[i, fiber_indices > -1] = \
          tangent[fiber_indices[fiber_indices > -1]]

    fod_samples, fod_weights = np.zeros((0, 3)), np.zeros(0)
    for i in range(nb_fibers):
        fod_samples = np.vstack((fod_samples, directions[i, compartments[i]]))
        fod_weights = np.hstack(
          (fod_weights, 1.0 / np.sum(compartments, 0)[compartments[i]]))
    
    return fod_samples, fod_weights / dim_grid


def compute_fod(fod_samples, fod_weights, dirs=None, kappa=30, sh=False, 
                order_sh=8):
    """
    Computes fod from a discrete set of weighted samples, using kernel density 
    estimation with a symmetric Von Mises-Fisher kernel.

    Parameters
    ----------
    fod_samples : array-like shape (P, 3)
        A discrete set of directions, as obtained by 
        :func:`compute_directions`.
    fod_weights : array-like shape (P, )
        The volume fractions associated to the directions, as obtained by 
        :func:`compute_directions`.
    dirs : array-like shape (M, 3)
        The directions on which to evaluate the fod. If None, uses a default 
        spherical 21-design (described in file ``spherical_21_design.txt``).
    kappa : double
        The concentration factor of the Von Mises-Fischer distribution.
    sh : bool
        If True, returns the spherical harmonic coefficients of the fod.
    order_sh : int
        Truncation order of the spherical harmonic representation.

    Returns
    -------
    fod : array-like, shape (M, )
        The fod, evaluated at the specified directions.
    """
    if dirs == None:
        __location__ = os.path.dirname(__file__)
        dirs = np.loadtxt(os.path.join(__location__, 
                          "./spherical_21_design.txt"))
    nb_samples = fod_samples.shape[0]
    nb_dirs = dirs.shape[0]
    fod = np.zeros(nb_dirs)
    c = kappa / (4 * np.pi * (np.exp(kappa) - 1))
    for i in range(nb_samples):
        dot_prods = np.dot(dirs, fod_samples[i])
        fod += np.exp(kappa * np.abs(dot_prods)) * fod_weights[i]
    fod *= c
    if not sh:
        return fod
    np.clip(dirs, -1, 1, dirs)
    x, y, z = dirs.T
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    H = shm.matrix(theta, phi, order=order_sh)
    pseudo_inv = np.dot(np.linalg.inv(np.dot(H.T, H)), H.T)
    res = np.dot(pseudo_inv, fod)
    return res


def _beta(kappa, n):
    r"""
    Computes the integral $\int_0^1 t^n \exp(\kappa t) \mathrm{d}t$.

    """
    if kappa == 0:
        return 1 / (n + 1)
    if n == 0:
        return (np.exp(kappa) - 1) / kappa
    else:
        beta_nm1 = _beta(kappa, n - 1)
        return (np.exp(kappa) - n*beta_nm1) / kappa


def c_vmf(kappa):
    """
    Computes the normalization constant of the Von-Mises Fischer kernel.
    """
    return kappa / (4 * np.pi * (np.exp(kappa) - 1))


def x_l(kappa, l):
    """
    Computes the l-th degree SH coefficient of the projection of the 
    symmetric Von-Mises Fischer kernel (centered at z-axis).

    Parameters
    ----------
    kappa : float
        The concentration parameter of the symmetric Von-Mises Fischer 
        distribution.
    l : int
        The degree of the SH coefficient to be computed.
    """
    a = kappa / (np.exp(kappa) - 1)
    a *= np.sqrt((2*l + 1) / (4 * np.pi))
    legendre_coeffs = legendre(l).coeffs[::-1]
    x = 0
    for n in range(0, l + 1, 2):
        x += legendre_coeffs[n] * _beta(kappa, n)
    return a*x


def compute_fod_sh(fod_samples, fod_weights, kappa=30, order_sh=8):
    """
    Computes fod using pure spherical harmonics from a discrete set of 
    weighted directions, using kernel density estimation with a symmetric Von 
    Mises-Fisher kernel.

    Parameters
    ----------
    fod_samples : array-like shape (P, 3)
        A discrete set of directions, as obtained by 
        :func:`compute_directions`.
    fod_weights : array-like shape (P, )
        The volume fractions associated to the directions, as obtained by 
        :func:`compute_directions`.
    kappa : double
        The concentration factor of the Von Mises-Fischer distribution.
    order_sh : int
        Truncation order of the spherical harmonic representation.

    Returns
    -------
    fod : array-like, shape (dim_sh, )
        The spherical harmonic coefficients of the FOD.
    """
    nb_samples = fod_samples.shape[0]
    if nb_samples == 0:
        return 0
    x, y, z = fod_samples.T
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    H = shm.matrix(theta, phi, order=order_sh)
    for l in range(0, order_sh + 1, 2):
        coeff_vmf = x_l(kappa, l)
        H[:, shm.dimension(l-2):shm.dimension(l)] *= coeff_vmf
    H *= fod_weights[:, np.newaxis]
    print H.shape
    return H.sum(0)
