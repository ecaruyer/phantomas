r"""This module contains code to generate synthetic diffusion signal
attenuation following several state-of-the-art model, including diffusion
tensor [1]_, stick model [2]_, hindered diffusion in cylinder [3]_, and
mixture of the above. All the models described here have axial and antipodal
symmetry, and we define an abstract class to serve as in interface.

References
----------
.. [1] Peter J. Basser, James Mattiello, and Denis LeBihan. "MR diffusion
   tensor spectroscopy and imaging." Biophysical journal 66, no. 1 (1994):
   259-267
.. [2] T. E. J. Behrens, M. W. Woolrich, M. Jenkinson, H. Johansen-Berg, R. G.
   Nunes, S. Clare, P. M. Matthews, J. M. Brady, and S. M. Smith.
   "Characterization and propagation of uncertainty in diffusion-weighted
   MR Imaging." Magnetic Resonance in Medicine 50, no. 5 (2003): 1077-1088.
.. [3] Soderman, Olle, and Bengt Jonsson. "Restricted diffusion in cylindrical
   geometry." Journal of Magnetic Resonance, Series A 117, no. 1 (1995):
   94-97.

"""
import numpy as np
from phantomas.utils import shm
from numpy.polynomial.legendre import Legendre


class AxiallySymmetricModel():
    """This class is an abstract class which defines the interface for all the
    synthetic models in use in phantomas.

    """

    def signal(self, bvals, thetas, bperps=None):
        r"""
        Returns the simulated signal attenuation. The angles thetas correspond
        to the angles between the sampling directions and the principal axis
        of the diffusion model. Must be implemented in subclasses.

        Parameters
        ----------
	bvals : array-like shape (K, )
            B-values [s\ mm\ :superscript:`-2`]. 
        thetas : array-like, shape (K, )
            Angles between the sampling directions and the axis.
        bperp : double
            b_perpendicular if we use cylindrical b-tensors [s\ mm\ :superscript:`-2`]. 

        """
        raise NotImplementedError("The method signal must be implemented in "
                                  "subclasses.")


    def odf(self, thetas, tau=1 / (4 * np.pi**2)):
        """Returns the ground truth ODF, when available.

        Parameters
        ----------
        thetas : array-like, shape (K, )
            Angles between the sampling directions and the axis.
        tau : double
            The diffusion time in s.

        """
        raise NotImplementedError("The method signal must be implemented in "
                                  "subclasses")


    def signal_convolution_sh(self, order, bval, nb_samples=100, bperp=None):
        r"""
        Returns the convolution operator in spherical harmonics basis, using
        the Funk-Hecke theorem as described in [1]_.

        Parameters
        ----------
        order : int
            The (even) spherical harmonics truncation order.
	bval : double
            B-value [s\ mm\ :superscript:`-2`].
        nb_samples : int
            The number of samples controling the accuracy of the numerical
            integral.
        bperp : double
            b_perpendicular if we use cylindrical b-tensors [s\ mm\ :superscript:`-2`]. 

        Note
        ----
        The function implemented here is the general, numerical implementation
        of the Funk-Hecke theorem. It is eventually replaced by analytical
        formula (when available) in subclasses.

        References
        ----------
        .. [1] Descoteaux, Maxime. "High angular resolution diffusion MRI: from
               local estimation to segmentation and tractography." PhD diss.,
               Universite de Nice Sophia-Antipolis, France, 2010.

        """
        cos_thetas = np.linspace(0, 1, nb_samples)
        thetas = np.arccos(cos_thetas)
        bvals = bval * np.ones(nb_samples)
        if bperp == None :
            fir = self.signal(bvals, thetas)
        else :
            bperps = bperp * np.ones(nb_samples)
            fir = self.signal(bvals, thetas, bperps=bperps)
        H = np.zeros((order + 1, nb_samples))
        dim_sh = shm.dimension(order)
        for l in range(0, order + 1, 2):
            coeffs = np.zeros(l + 1)
            coeffs[l] = 1.0
            H[l, :] = Legendre(coeffs)(cos_thetas)
        ls = list(map(shm.sh_degree, range(dim_sh)))
        rs = np.dot(H, fir) / nb_samples
        return rs[ls]


class GaussianModel(AxiallySymmetricModel):
    r"""
    This class models a Gaussian diffusion tensor wot axial symmetry.
    Typically, the eigenvalues of this tensors are
    :math:`\lambda_1 \gg \lambda_2 = \lambda_3`.

    Parameters
    ----------
    lambda1 : double
        The eigenvalue associated with the principal direction, in
        mm\ :sup:`2`/s.
    lambda2 : double
        The eigenvalue associated with the two minor eigenvectors, in
        mm\ :sup:`2`/s.
    """

    def __init__(self, lambda1=1.7e-3, lambda2=0.2e-3):
        self.lambda1 = lambda1
        self.lambda2 = lambda2


    def signal(self, bvals, thetas, bperps=None):
        r"""Returns the simulated signal attenuation, following the Stejskal
        and Tanner [1]_ equation. The angles thetas correspond to the angles
        between the sampling directions and the principal axis of the
        diffusion tensor.

        Parameters
        ----------
	bvals : array-like shape (K, )
            B-values [s\ mm\ :superscript:`-2`]. 
        thetas : array-like, shape (K, )
            Angles between the sampling directions and the axis.
        bperps : array-like shape (K, )
            b_perpendicular if we use cylindrical b-tensors [s\ mm\ :superscript:`-2`]. 

        """

        if bperps is None :
            signal = np.exp(-bvals * self.lambda2)
            signal *= np.exp(-bvals * (self.lambda1 - self.lambda2) \
                             * np.cos(thetas)**2)
        else :
            signal = np.exp(-(bvals - 3 * bperps) * (self.lambda1 - self.lambda2) \
                            * np.cos(thetas)**2)
            signal *= np.exp(-(bvals - 3 * bperps) * self.lambda2)
            signal *= np.exp(-bperps * (self.lambda1 - self.lambda2))
            signal *= np.exp(-3 * bperps * self.lambda2)

        return signal
