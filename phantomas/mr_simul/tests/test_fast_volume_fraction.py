from phantomas.mr_simul.fast_volume_fraction import *
import unittest
import numpy as np

from nose.tools import (assert_true, assert_equal, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_, assert_almost_equal)


def test_compute_fod():
    nb_directions = 10
    theta = np.arccos(np.random.uniform(-1, 1, nb_directions))
    phi = np.random.uniform(0, np.pi, nb_directions)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    directions = np.vstack((x, y, z)).T
    directions = np.ascontiguousarray(directions)
    weights = np.ones(nb_directions)
    fod = compute_fod(directions, weights)
