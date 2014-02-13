"""
This module contains functions for MR image formation, such as random
generation of T1/T2 relaxation time images, etc. In this regard, the mean and
standard deviation of relaxation times of biological tissues, are taken from
[1]_.

References
----------
.. [1] Wansapura, Janaka P., Scott K. Holland, R. Scott Dunn, and William S.
   Ball. "NMR relaxation times in the human brain at 3.0 tesla." Journal of
   magnetic resonance imaging 9, no.  4 (1999): 531-538.
"""
import numpy as np
import scipy.sparse as scisp
from scikits.sparse.cholmod import cholesky


def _random_correlated_image(mean, sigma, image_shape, alpha=0.3, seed=None):
    """
    Creates a random image with correlated neighbors.
    pixel covariance is sigma^2, direct neighors pixel covariance is alpha * sigma^2.

    Parameters
    ----------
    mean : the mean value of the image pixel values.
    sigma : the std dev of image pixel values.
    image_shape : tuple, shape = (3, )
    alpha : the neighbors correlation factor.
    seed : the seed to use for the random number generator, default : None
    """
    dim_x, dim_y, dim_z = image_shape
    dim_image = dim_x * dim_y * dim_z

    correlated_image = 0
    for neighbor in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        corr_data = []
        corr_i = []
        corr_j = []
        for i, j, k in [(0, 0, 0), neighbor]:
            d2 = 1.0 * (i*i + j*j + k*k)
            ind = np.asarray(np.mgrid[0:dim_x-i, 0:dim_y-j, 0:dim_z-k], dtype=np.int)
            ind = ind.reshape((3, (dim_x - i) * (dim_y - j) * (dim_z - k)))
            corr_i.extend(np.ravel_multi_index(ind, (dim_x, dim_y, dim_z)).tolist())
            corr_j.extend(np.ravel_multi_index(ind + np.asarray([i, j, k])[:, None],
                                          (dim_x, dim_y, dim_z)).tolist())
            if i>0 or j>0 or k>0:
                corr_i.extend(np.ravel_multi_index(ind + np.asarray([i, j, k])[:, None],
                                              (dim_x, dim_y, dim_z)).tolist())
                corr_j.extend(np.ravel_multi_index(ind, (dim_x, dim_y, dim_z)).tolist())
            if i==0 and j==0 and k==0:
                corr_data.extend([3.0] * ind.shape[1])
            else:
                corr_data.extend([alpha * 3.0] * 2 * ind.shape[1])

        correlation = scisp.csc_matrix((corr_data, (corr_i, corr_j)), shape=(dim_image, dim_image))

        factor = cholesky(correlation)
        L = factor.L()
        P = factor.P()[None, :]
        P = scisp.csc_matrix((np.ones(dim_image),
                              np.vstack((P, np.asarray(range(dim_image))[None, :]))),
                             shape=(dim_image, dim_image))

        sq_correlation = P.dot(L)

        RNG = np.random.RandomState(seed)
        X = RNG.normal(0, 1, dim_image)
        Y = sq_correlation.dot(X)
        Y = Y.reshape((dim_x, dim_y, dim_z))
        X = X.reshape((dim_x, dim_y, dim_z))
        correlated_image += Y
    correlated_image /= 3

    return correlated_image * sigma + mean

_physical_parameters = {
    'wm' : {
        't1' : {'mean' : 0.832,   'stddev' : 0.010},
        't2' : {'mean' : 79.6e-3, 'stddev' : 0.6e-3,},
        'rho' : 0.65,
    },
    'gm' : {
        't1' : {'mean' : 1.331,   'stddev' : 0.013,},
        't2' : {'mean' : 110.e-3, 'stddev' : 2.0e-3,},
        'rho' : 0.75,
    },
    'csf' : {
        't1' : {'mean' : 3.5,     'stddev' : 0.1,},
        't2' : {'mean' : 0.25,    'stddev' : 0.01,},
        'rho' : 1.0,
    },
}


def relaxation_time_images(image_shape, tissue_type, seed=None):
    """
    Return randomly generated images of t1 and t2 relaxation times, of
    desired shape, for the desired tissue type.

    Parameters
    ----------
    image_shape : tuple
        ``dim_x, dim_y, dim_z``
    tissue_type : 'wm', 'gm', 'csf'
        The tissue type, either white matter (WM), gray matter (GM), or
        cerebro-spinal fluid (CSF).

    Returns
    -------
    t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        T1 relaxation time image.
    t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        T2 relaxation time image.
    """
    t1 = _random_correlated_image(_physical_parameters[tissue_type]['t1']['mean'],
                                 _physical_parameters[tissue_type]['t1']['stddev'],
                                 image_shape, seed=seed)
    t2 = _random_correlated_image(_physical_parameters[tissue_type]['t2']['mean'],
                                 _physical_parameters[tissue_type]['t2']['stddev'],
                                 image_shape, seed=seed)
    return t1, t2


def mr_signal(wm_vf, wm_t1, wm_t2,
              gm_vf, gm_t1, gm_t2,
              csf_vf, csf_t1, csf_t2,
              te, tr):
    """
    Computes MR image, provided images of the WM, GM, CSF and background volume
    fractions.

    Parameters
    ----------
    wm_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        White matter volume fraction.
    wm_t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        White matter t1 relaxation image.
    wm_t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        White matter t2 relaxation image.
    gm_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        Gray matter volume fraction
    gm_t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        Gray matter t1 relaxation image.
    gm_t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        Gray matter t2 relaxation image.
    csf_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        CSF volume fraction
    csf_t1 : array-like, shape ``(dim_x, dim_y, dim_z)``
        CSF t1 relaxation image.
    csf_t2 : array-like, shape ``(dim_x, dim_y, dim_z)``
        CSF t2 relaxation image.
    background_vf : array-like, shape ``(dim_x, dim_y, dim_z)``
        Background volume fraction
    te : double
        echo time (s)
    tr : double
        repetition time (s)

    Returns
    -------
    image : array-like, shape ``(dim_x, dim_y, dim_z)``
        The computed MR signal.
    """
    wm_rho = _physical_parameters['wm']['rho']
    gm_rho = _physical_parameters['gm']['rho']
    csf_rho = _physical_parameters['csf']['rho']
    image_shape = wm_vf.shape
    image = np.zeros(image_shape)
    image += wm_vf * wm_rho * (1.0 - np.exp(-tr / wm_t1)) * np.exp(-te / wm_t2)
    image += gm_vf * gm_rho * (1.0 - np.exp(-tr / gm_t1)) * np.exp(-te / gm_t2)
    image += csf_vf * csf_rho * (1.0 - np.exp(-tr / csf_t1)) * np.exp(-te / csf_t2)

    return image


def rician_noise(image, sigma, seed1=None, seed2=None):
    """
    Add Rician distributed noise to the input image.

    Parameters
    ----------
    image : array-like, shape ``(dim_x, dim_y, dim_z)`` or ``(dim_x, dim_y,
        dim_z, K)``
    sigma : double
    seed1 : the seed to use for the random number generator of the
        first gaussian, default : None
    seed2 : the seed to use for the random number generator of the
        second gaussian, default : None
    """
    RNG1 = np.random.RandomState(seed1)
    RNG2 = np.random.RandomState(seed2)
    n1 = RNG1.normal(loc=0, scale=sigma, size=image.shape)
    n2 = RNG2.normal(loc=0, scale=sigma, size=image.shape)
    return np.sqrt((image + n1)**2 + n2**2)
