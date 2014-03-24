"""
Model definition for the fibers geometry. A ``Fiber`` is a discrete
representation of a white matter fiber bundle.
"""
import numpy as np


class Fiber():
    """
    A Fiber is a cylindrical shape wrapped around a discrete curve in 3D,
    represented by its discretization over a certain number of points.

    Parameters
    ----------
    mode : 'from_points'
    points : array-like, shape (N, 3)
        A sequence of points representing the fiber. These may have been from
        a ``FiberSource``.
    tangents : array-like, shape (N, 3)
        The tangents to the fiber bundle at the specified points.
    radius : double
        The radius of the fiber bundle in mm.
    """

    def __init__(self, mode, **kwargs):
        if mode == 'from_points':
            points = kwargs.get('points')
            tangents = kwargs.get('tangents', None)
            radius = kwargs.get('radius', 1.0)
            self._create_from_points(points, tangents, radius)

    def _create_from_points(self, points, tangents, radius):
        """
        Creates a Fiber object from fiber positions, tangents and radius.
        These the points and tangents are typically obtained from a
        FiberSource object.

        Parameters
        ----------
        points : array-like shape (N, 3)
        tangents : array-like shape (N, 3)
        """
        self.points = points
        self.tangents = tangents
        self.nb_points = points.shape[0]
        self.radius = radius

    def set_radius(self, radius):
        """
        Sets the radius of the fiber bundle.

        Parameters
        ----------
        radius : double
        """
        self.radius = radius

    def get_radius(self):
        """
        Gets the radius of the fiber bundle.

        Returns
        -------
        radius : double
        """
        return self.radius

    def get_points(self):
        """
        Gets the points of the fiber bundle centerline.

        Returns
        -------
        points : array-like, shape (N, 3)
        """
        return self.points

    def get_nb_points(self):
        """
        Gets the number of points over which the center line is defined.

        Returns
        -------
        nb_points : int
        """
        return self.nb_points

    def save_to_file(self, index, path='.'):
        """
        Saves the fiber trajectory to a text file.

        Parameters
        ----------
        index : int
            The identifier of the fiber bundle (will be used to format the
            filename).
        path : string
            The output path.
        """
        file_name = '%s/fiber_%02d_r%.6f.txt' % (path, index, self.radius)
        savetxt(fileName, self.points)

    def intersects_bounding_box(self, bounding_box_extents):
        """
        Computes logical intersection between current instance and given
        bounding box.

        Parameters
        ----------
        bounding_box_extents : tuple
            (x_min, x_max, y_min, y_max, z_min, z_max)

        Returns
        -------
        intersects : bool
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bounding_box_extents
        nb_points = self.get_nb_points()
        radius = self.get_radius()
        points = self.get_points()

        matching_points = np.logical_and(points[:, 0] > x_min - radius,
                                         points[:, 0] < x_max + radius)
        matching_points = np.logical_and(matching_points,
                                         points[:, 1] > y_min - radius)
        matching_points = np.logical_and(matching_points,
                                         points[:, 1] < y_max + radius)
        matching_points = np.logical_and(matching_points,
                                         points[:, 2] > z_min - radius)
        matching_points = np.logical_and(matching_points,
                                         points[:, 2] < z_max + radius)

        return np.count_nonzero(matching_points) > 0


class IsotropicRegion():
    """
    An ``IsotropicRegion`` is a spherical region defined by its radius and
    center. It usually defines a region of rapid diffusivity, modelling a
    cerebro-spinal fluid-filled region, such as ventricle.

    Parameters
    ----------
    radius : double
    center : array-like, shape (3, )
        The center of the spherical region.
    """

    def __init__(self, radius, center, volume_fraction):
        self.radius = radius
        self.center = center
        self.volume_fraction = volume_fraction

    def get_radius(self):
        """
        Returns
        -------
        radius : double
            The radius of the ``IsotropicRegion``, in mm.

        """
        return self.radius

    def get_center(self):
        """
        Returns
        -------
        center : array-like, shape (3, )
            The center position of the ``IsotropicRegion``, in real-world
            coordinates.

        """
        return self.center

    def get_volume_fraction(self):
        """
        Returns
        -------
        volume_fraction : double
            The water volume fraction.

        """
        return self.volume_fraction

