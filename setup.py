from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os


setup(name='Phantomas',
      description='A software phantom generation tool for diffusion MRI.',
      version='0.1.dev',
      author='Emmanuel Caruyer',
      author_email='caruyer@gmail.com',
      url='http://www.emmanuelcaruyer.com/phantomas/',
      packages=['phantomas', 
                'phantomas.geometry', 
                'phantomas.mr_simul', 
                'phantomas.utils',
                'phantomas.visu'],
      package_data={'phantomas.mr_simul' : ["spherical_21_design.txt"]},
      scripts = [os.path.join('scripts', 'phantomas_struct'),
                 os.path.join('scripts', 'phantomas_dwis'),
                 os.path.join('scripts', 'phantomas_rois'),
                 os.path.join('scripts', 'phantomas_masks'),
                 os.path.join('scripts', 'phantomas_view'),
                 os.path.join('scripts', 'sticks2dwis')],
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("phantomas.mr_simul.fast_volume_fraction",
                             sources=["phantomas/mr_simul/fast_volume_fraction.pyx",
                                      "phantomas/mr_simul/c_fast_volume_fraction.c"],
                             include_dirs=[np.get_include(), "/usr/local/include/"],
                             library_dirs=["/usr/lib/"],
                             libraries=["gsl", "gslcblas"])],
      )
