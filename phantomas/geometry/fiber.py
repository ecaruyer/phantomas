"""
This module contains the definition of ``FiberSource``, which is a continuous
representation of a fiber. All the fibers created are supposed to connect two
cortical areas. Currently, the only supported shape for the "cortical surface"
is a sphere.
"""
import numpy as np
try:
    from scipy.interpolate import PiecewisePolynomial
except ImportError:
    from scipy.interpolate import PPoly as PiecewisePolynomial


class FiberSource:
    """
    A ``FiberSource`` is a continuous representation of a fiber trajectory,

    .. math::
        f: [0, 1] \\rightarrow \\mathcal{R}^3.  

    The trajectory is modeled by 3 piecewise polynomials (one for each 
    dimension). Note that the fiber is connecting two end points on the 
    "cortical surface". The construction makes sure that the tangents to the 
    fiber are normal to this surface.

    Parameters
    ----------
    control_points : array-like shape (nb_points, 3) 
        a set of points through which the fiber will go.
    tangents : optional, default = 'symmetric'
        Either 'symmetric', 'incoming', 'outgoing'. Controls the way the 
        tangents are computed.
    scale : optional, default = 1.0
        A multiplicative factor for the points positions. This corresponds to 
        the sphere radius in mm.
    """
    def __init__(self, control_points, **kwargs):
        # The mode to compute tangents, either 'symmetric', 'incoming', or 
        # 'outgoing'
        tangents = kwargs.get('tangents', 'symmetric')
        scale = kwargs.get('scale', 1.0)
        self._create_from_control_points(control_points, tangents, scale)


    def __call__(self, ts):
        return self.interpolate(ts)


    def _create_from_control_points(self, control_points, tangents, scale):
        """
        Creates the FiberSource instance from control points, and a specified 
        mode to compute the tangents.

        Parameters
        ----------
        control_points : ndarray shape (N, 3)
        tangents : 'incoming', 'outgoing', 'symmetric'
        scale : multiplication factor. 
            This is useful when the coodinates are given dimensionless, and we 
            want a specific size for the phantom.
        """
        # Compute instant points ts, from 0. to 1. 
        # (time interval proportional to distance between control points)
        nb_points = control_points.shape[0]
        dists = np.zeros(nb_points)
        dists[1:] = np.sqrt((np.diff(control_points, axis=0) ** 2).sum(1))
        ts = dists.cumsum()
        length = ts[-1]
        ts = ts / np.max(ts)

        # Create interpolation functions (piecewise polynomials) for x, y and z
        derivatives = np.zeros((nb_points, 3))

        # The derivatives at starting and ending points are normal
        # to the surface of a sphere.
        derivatives[0, :] = -control_points[0]
        derivatives[-1, :] = control_points[-1]
 
        # As for other derivatives, we use discrete approx
        if tangents == 'incoming':
            derivatives[1:-1, :] = (control_points[1:-1] - control_points[:-2])
        elif tangents == 'outgoing':
            derivatives[1:-1, :] = (control_points[2:] - control_points[1:-1])
        elif tangents == 'symmetric':
            derivatives[1:-1, :] = (control_points[2:] - control_points[:-2])
        else:
            raise Error('tangents should be one of the following: incoming, ' 
                        'outgoing, symmetric')
 
        derivatives = (derivatives.T / np.sqrt((derivatives ** 2).sum(1))).T \
                    * length
               
        self.x_poly = PiecewisePolynomial(ts, 
               scale * np.vstack((control_points[:, 0], derivatives[:, 0])).T)
        self.y_poly = PiecewisePolynomial(ts, 
               scale * np.vstack((control_points[:, 1], derivatives[:, 1])).T)
        self.z_poly = PiecewisePolynomial(ts, 
               scale * np.vstack((control_points[:, 2], derivatives[:, 2])).T)


    def interpolate(self, ts):
        """
        From a ``FiberSource``, which is a continuous representation, to a 
        ``Fiber``, a discretization of the fiber trajectory.

        Parameters
        ----------
        ts : array-like, shape (N, ) 
            A list of "timesteps" between 0 and 1.

        Returns
        -------
        trajectory : array-like, shape (N, 3)
            The trajectory of the fiber, discretized over the provided 
            timesteps.
        """
        N = ts.shape[0]
        trajectory = np.zeros((N, 3))
        trajectory[:, 0] = self.x_poly(ts)
        trajectory[:, 1] = self.y_poly(ts)
        trajectory[:, 2] = self.z_poly(ts)
        return trajectory


    def tangents(self, ts):
        """
        Get tangents (as unit vectors) at given timesteps.

        Parameters
        ----------
        ts : array-like, shape (N, ) 
            A list of "timesteps" between 0 and 1.

        Returns
        -------
        tangents : array-like, shape (N, 3)
            The tangents (as unit vectors) to the fiber at selected timesteps.
        """
        x_der = self.x_poly.derivative(ts, der=1)
        y_der = self.y_poly.derivative(ts, der=1)
        z_der = self.z_poly.derivative(ts, der=1)
        N = ts.shape[0]
        tangents = np.zeros((N, 3))
        tangents[:, 0] = x_der
        tangents[:, 1] = y_der
        tangents[:, 2] = z_der
        tangents = tangents / np.sqrt(np.sum(tangents ** 2, 1))[:, np.newaxis]
        return tangents


    def curvature(self, ts):
        """
        Evaluates the curvature of the fiber at given positions. The curvature 
        is computed with the formula
        
        .. math::
            \gamma = \\frac{\|f'\wedge f''\|}{\|f'\|^3}\qquad.
 
        Parameters
        ----------
        ts : array-like, shape (N, ) 
            A list of "timesteps" between 0 and 1.

        Returns
        -------
        curvatures : array-like, shape (N, )
            The curvatures of the fiber trajectory, at selected timesteps.
        """
        x_der1 = self.x_poly.derivative(ts, der=1)
        x_der2 = self.x_poly.derivative(ts, der=2)
        y_der1 = self.y_poly.derivative(ts, der=1)
        y_der2 = self.y_poly.derivative(ts, der=2)
        z_der1 = self.z_poly.derivative(ts, der=1)
        z_der2 = self.z_poly.derivative(ts, der=2)
        curv  = (z_der2*y_der1 - y_der2*z_der1)**2
        curv += (x_der2*z_der1 - z_der2*x_der1)**2
        curv += (y_der2*x_der1 - x_der2*y_der1)**2
        curv /= (x_der1**2 + y_der1**2 + z_der1**2)**3
        np.sqrt(curv, out=curv)
        return curv
