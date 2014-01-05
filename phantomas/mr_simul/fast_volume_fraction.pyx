import cython
from libc.stdlib cimport malloc, free

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


cdef extern void c_in_fiber(double* points, int nb_points,
    double* fiber_points, double* fiber_tangents, int fiber_nb_points,
    double fiber_radius, int* segment_indices)


def in_fiber(np.ndarray[double, ndim=2, mode="c"] points,
             np.ndarray[double, ndim=2, mode="c"] fiber_points,
             np.ndarray[double, ndim=2, mode="c"] fiber_tangents,
             double fiber_radius):
    """
    Checks whether a collection of points lie within a cylindrical fiber bundle.
    This returns an array of the same lenght as points, corresponding
    to the corresponding fiber segment index.

    Parameters
    ----------
    points : ndarray (N, 3)
    fiber_points : ndarray (M, 3)
    fiber_tangents : ndarray (M, 3)
    fiber_radius : double
    """
    cdef int nb_points = points.shape[0]
    cdef int fiber_nb_points = fiber_points.shape[0]
    cdef np.ndarray[int, ndim=1] segment_indices

    segment_indices = -np.ones(nb_points, dtype=np.intc)

    c_in_fiber(&points[0, 0], nb_points, 
               &fiber_points[0, 0], &fiber_tangents[0, 0], fiber_nb_points,
               fiber_radius, &segment_indices[0])

    return segment_indices
