"""
This package contains implementation of the simulation of MR images and MR 
diffusion signal attenuation. The basic pipeline is as follows: 

1. We first generate random images of T1, T2 relaxation times, for each tissue
   type (white matter, gray matter and cerebro-spinal fluid). See 
   :func:`image_formation.relaxation_time_images`.
2. From a description of the fiber bundles, we compute the local diffusion
   directions on a subdivision of the image grid. See
   :func:`fod.compute_directions`.
3. Based on these diffusion directions, we compute a continuous FOD, using
   kernel density estimation. See :func:`fod.compute_fod`.
4. Using a synthetic model of diffusion, we copmute the signal attenuation for
   the desired samples in Q-space, using convolution of the FOD by a fiber 
   impulse response. See 
   :func:`synthetic.AxiallySymmetricModel.signal` and
   :func:`synthetic.AxiallySymmetricModel.signal_convolution_sh`.
"""

__all__ = ['fast_volume_fraction', 'image_formation', 'partial_volume', 'fod', 
           'synthetic']

