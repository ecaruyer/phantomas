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


class SphericalHarmonics:
    """This class describes a symmetrical spherical function by its spherical 
    hamonics coefficients.

    Parameters
    ----------
    coefficients : array-like, shape (R, )
        The coefficients vector of the spherical harmonics function. The order
        in which the coefficients are stored is described in :func:`j`.
    """
    
    def __init__(self, coefficients):
        self._create_from_coefficients(coefficients)

    
    def _create_from_coefficients(self, coefficients):
        order = 2
        while True:
            dimension = (order + 1) * (order + 2) / 2
            if len(coefficients) == dimension:
                self.order = order
                self.coefficients = coefficients
                return
            elif len(coefficients) < dimension:
                raise NameError("Invalid dimension for SH coefficients.")
            order += 2


    def set_coefficients(self, coefficients):
        self.coefficients[:] = coefficients[:]


    def angular_function(self, theta, phi):
        """
        Returns the values of the spherical harmonics function at given 
        positions specified by colatitude and aximuthal angles.

        Parameters
        ----------
        theta : array-like, shape (K, )
            The colatitude angles.
        phi : array0-like, shape (K, )
            The azimuth angles.

        Returns
        -------
        f : array-like, shape (K, )
            The value of the function at given positions.
        """
        coefs = self.coefficients
        result = 0
        order = self.order
        j = 0
        # We follow here reverse convention about theta and phi w.r.t scipy.
        for l in range(0, order+1, 2):
            for m in range(-l, l+1):
                sh = sph_harm(np.abs(m), l, phi, theta)
                if coefs[j] != 0.0:
                    if m < 0:
                        result += coefs[j] * np.sqrt(2) * sh.real
                    if m == 0:
                        result += coefs[j] * sh.real
                    if m > 0:
                        result += coefs[j] * np.sqrt(2) * sh.imag
                j = j+1
        return result


    def spherical_function(self, x, y, z):
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
        return self.angular_function(theta, phi)


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


def l(j):
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


def m(j):
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
    l = l(j)
    return j + l + 1 - dimension(l)


def matrix(theta, phi, order=4):
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

    Returns
    -------
    H : array-like, shape (K, R)
        The spherical harmonics observation matrix.
    """
    
    dim_sh = dimension(order)
    sh = SphericalHarmonics(np.zeros(dim_sh))
    N = theta.shape[0]
    H = np.zeros((N, dim_sh))
    for j in range(dim_sh):
        sh.coefficients[:] = 0
        sh.coefficients[j] = 1.0
        H[:, j] = sh.angular_function(theta, phi)
    return H


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
        l =  l(j)
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
        l =  l(j)
        P[j, j] = 2 * pi * legendre(l)(0)
    return P
