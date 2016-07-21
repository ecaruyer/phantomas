"""
This module contains an implementation of the real, antipodally symmetric
Spherical Harmonics basis as defined in [1]_.

References
----------
.. [1] Descoteaux, Maxime, Elaine Angelino, Shaun Fitzgibbons, and Rachid
   Deriche. "Regularized, fast, and robust analytical Q-ball imaging" 
   Magnetic Resonance in Medicine 58, no. 3 (2007): 497-510

"""
import numpy as np
from scipy.misc import factorial
from scipy.special import lpmv, legendre, sph_harm
import hashlib


def angular_function(j, theta, phi):
    """
    Returns the values of the spherical harmonics function at given 
    positions specified by colatitude and aximuthal angles.

    Parameters
    ----------
    j : int
        The spherical harmonic index.
    theta : array-like, shape (K, )
        The colatitude angles.
    phi : array-like, shape (K, )
        The azimuth angles.

    Returns
    -------
    f : array-like, shape (K, )
        The value of the function at given positions.
    """
    l = sh_degree(j)
    m = sh_order(j)
    # We follow here reverse convention about theta and phi w.r.t scipy.
    sh = sph_harm(np.abs(m), l, phi, theta)
    if m < 0:
        return np.sqrt(2) * sh.real
    if m == 0:
        return sh.real
    if m > 0:
        return np.sqrt(2) * sh.imag


def spherical_function(j, x, y, z):
    """
    Returns the values of the spherical harmonics function at given 
    positions specified by Cartesian coordinates.

    Parameters
    ----------
    x, y, z : array-like, shape (K, )
        Cartesian coordinates.

    Returns
    -------
    f : array-like, shape (K, )
        The value of the function at given positions.
    """
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return angular_function(j, theta, phi)


def dimension(order):
    r"""
    Returns the dimension, :math:`R`, of the real, antipodally symmetric 
    spherical harmonics basis for a given truncation order.
    
    Parameters
    ----------
    order : int
        The trunction order.
    
    Returns
    -------
    R : int
        The dimension of the truncated spherical harmonics basis.
    """
    return (order + 1) * (order + 2) / 2


def j(l, m):
    r"""
    Returns the flattened spherical harmonics index corresponding to degree
    ``l`` and order ``m``.

    Parameters
    ----------
    l : int
        Degree of the spherical harmonics. Should be even.
    m : int
        Order of the spherical harmonics, should verify :math:`-l \leq m \leq l`

    Returns
    -------
    j : int
        The associated index of the spherical harmonic.
    """
    if np.abs(m) > l:
        raise NameError('SphericalHarmonics.j: m must lie in [-l, l]')
    return int(l + m + (2 * np.array(range(0, l, 2)) + 1).sum())


def sh_degree(j):
    """
    Returns the degree, ``l``, of the spherical harmonic associated to index 
    ``j``.

    Parameters
    ----------
    j : int
        The flattened index of the spherical harmonic.

    Returns
    -------
    l : int
        The associated even degree.
    """
    l = 0
    while dimension(l) - 1 < j:
        l += 2
    return l


def sh_order(j):
    """
    Returns the order, ``m``, of the spherical harmonic associated to index 
    ``j``.

    Parameters
    ----------
    j : int
        The flattened index of the spherical harmonic.

    Returns
    -------
    m : int
        The associated order.
    """
    l = sh_degree(j)
    return j + l + 1 - dimension(l)


class _CachedMatrix():
    """
    Returns the spherical harmonics observation matrix.

    Parameters
    ----------
    theta : array-like, shape (K, )
        The colatitude angles.
    phi : array-like, shape (K, )
        The azimuth angles.
    order : int
        The spherical harmonics truncation order.
    cache : bool
        Whether the result should be cached or not.

    Returns
    -------
    H : array-like, shape (K, R)
        The spherical harmonics observation matrix.
    """
    def __init__(self):
        self._cache = {}


    def __call__(self, theta, phi, order=4, cache=True):
        if not cache:
            return self._eval_matrix(theta, phi, order)
        key1 = self._hash(theta)
        key2 = self._hash(phi)
        if (key1, key2, order) in self._cache:
            return self._cache[(key1, key2, order)]
        else:
            val = self._eval_matrix(theta, phi, order)
            self._cache[(key1, key2, order)] =  val
            return val
        

    def _hash(self, np_array):
        return hashlib.sha1(np_array).hexdigest()

   
    def _eval_matrix(self, theta, phi, order):
        N = theta.shape[0]
        dim_sh = dimension(order)
        ls = [l for L in range(0, order + 1, 2) for l in [L] * (2*L + 1)]
        ms = [m for L in range(0, order + 1, 2) for m in range(-L, L+1)]
        ls = np.asarray(ls, dtype=np.int)[np.newaxis, :]
        ms = np.asarray(ms, dtype=np.int)[np.newaxis, :]
        sh = sph_harm(np.abs(ms), ls, 
                      phi[:, np.newaxis], theta[:, np.newaxis])
        H = np.where(ms > 0, sh.imag, sh.real)
        H[:, (ms != 0)[0]] *= np.sqrt(2)
        return H

matrix = _CachedMatrix()


def L(order=4):
    """Computees the Laplace-Beltrami operator matrix.

    Parameters
    ----------
    order : int
        The truncation order (should be an even number).
    """
    dim_sh = dimension(order)
    L = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l =  sh_degree(j)
        L[j, j] = - (l * (l + 1))
    return L


def P(order=4):
    """Returns the Funk-Radon operator matrix.

    Parameters
    ----------
    order : int
        The truncation order (should be an even number).
    """
    dim_sh = dimension(order)
    P = zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l =  sh_degree(j)
        P[j, j] = 2 * pi * legendre(l)(0)
    return P


def convert_to_mrtrix(order):
    """
    Returns the linear matrix used to convert coefficients into the mrtrix 
    convention for spherical harmonics.

    Parameters
    ----------
    order : int

    Returns
    -------
    conversion_matrix : array-like, shape (dim_sh, dim_sh)
    """
    dim_sh = dimension(order)
    conversion_matrix = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l = sh_degree(j)
        m = sh_order(j)
        if m == 0:
            conversion_matrix[j, j] = 1
        else:
            conversion_matrix[j, j - 2*m] = np.sqrt(2)
    return conversion_matrix
