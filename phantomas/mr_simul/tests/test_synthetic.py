from phantomas.mr_simul.synthetic import *
import unittest
import numpy as np

from nose.tools import (assert_true, assert_equal, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_, assert_almost_equal, run_module_suite)


def test_signal():
    model = GaussianModel()
    nb_directions = 100
    theta = np.arccos(np.random.uniform(-1, 1, nb_directions))
    nb_samples = 100
    for bval in np.random.uniform(0, 10000, 30):
        bvals = bval * np.ones(nb_samples)
        bvecs = 0 * np.ones(nb_samples)
        assert_almost_equal(model.signal(bvals, theta, bvecs), model.signal(bvals, theta))

def test_signal_spherical_b_tensor():
    model = GaussianModel()
    nb_directions = 100
    theta = np.arccos(np.random.uniform(-1, 1, nb_directions))
    nb_samples = 100
    theta2 = np.arccos(np.random.uniform(-1, 1, nb_directions))
    for bval in np.random.uniform(0, 10000, 30):
        bvals = bval * np.ones(nb_samples)
        bperps = bvals / 3
        assert_almost_equal(model.signal(bvals, theta, bperps), model.signal(bvals, theta2, bperps))

def test_signal_spherical_fiber_tensor():
    model = GaussianModel(1.7e-3, 1.7e-3)
    nb_directions = 100
    theta = np.arccos(np.random.uniform(-1, 1, nb_directions))
    nb_samples = 100
    theta2 = np.arccos(np.random.uniform(-1, 1, nb_directions))
    for bval in np.random.uniform(0, 10000, 30):
        for bperp in np.random.uniform(0, 10000, 30):
            bvals = bval * np.ones(nb_samples)
            bperps = bperp * np.ones(nb_samples)
            assert_almost_equal(model.signal(bvals, theta, bperps), model.signal(bvals, theta2, bperps))


if __name__ == '__main__':
    run_module_suite()
