#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def main():
    from glob import glob
    from setuptools import setup, find_packages
    from setuptools.extension import Extension

    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import numpy as np

    REQ_LINKS = []
    with open('requirements.txt', 'r') as rfile:
        REQUIREMENTS = [line.strip() for line in rfile.readlines()]

    for i, req in enumerate(REQUIREMENTS):
        if req.startswith('-e'):
            REQUIREMENTS[i] = req.split('=')[1]
            REQ_LINKS.append(req.split()[1])

    if REQUIREMENTS is None:
        REQUIREMENTS = []

    extensions = [Extension(
            "phantomas.mr_simul.fast_volume_fraction",
            ["phantomas/mr_simul/fast_volume_fraction.pyx",
            "phantomas/mr_simul/c_fast_volume_fraction.c"],
            include_dirs=[np.get_include(), "/usr/local/include/"],
            library_dirs=["/usr/lib/"],
            libraries=["gsl", "gslcblas"]),
    ]

    setup(name='phantomas',
          description='A software phantom generation tool for diffusion MRI.',
          version='0.1.dev',
          author='Emmanuel Caruyer',
          author_email='caruyer@gmail.com',
          url='http://www.emmanuelcaruyer.com/phantomas/',
          install_requires=REQUIREMENTS,
          dependency_links=REQ_LINKS,
          packages=find_packages(),
          package_data={'phantomas.mr_simul': ["spherical_21_design.txt"]},
          scripts=glob('scripts/*'),
          # cmdclass={'build_ext': build_ext},
          ext_modules=cythonize(extensions),
    )

if __name__ == '__main__':
    LOCAL_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(LOCAL_PATH)
    sys.path.insert(0, LOCAL_PATH)

    main()

