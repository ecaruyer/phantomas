#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

__packagename__ = 'phantomas'
__version__ = '0.1.dev'
__author__ = 'Emmanuel Caruyer'
__credits__ = ['Emmanuel Caruyer']
__license__ = '3-clause BSD'
__maintainer__ = 'Emmanuel Caruyer'
__email__ = 'caruyer@gmail.com'
__status__ = 'Prototype'
__copyright__ = 'Copyright {}, {}'.format(datetime.now().year, __author__)

__description__ = """A software phantom generation tool for diffusion MRI."""
__longdesc__ = """\
*Phantomas* is an open-source software library for the creation of *realistic \
phantoms* in *diffusion MRI*. This is intented as a tool for the quantitative \
evaluation of methods in acquisition, signal and image processing, local \
reconstruction and fiber tracking in diffusion MRI. Phantomas is released \
under the terms of the revised BSD license. You should have received a copy \
of the license together with the software, please refer to the file \
``LICENSE``.
"""

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 2.7',
]

DOWNLOAD_URL = 'https://github.com/ecaruyer/phantomas'
URL = 'http://www.emmanuelcaruyer.com/{}'.format(__packagename__)

# Dependencies of phantomas
REQUIRES = [
    'cython',
    'nibabel',
    'numpy',
    'scikits.sparse',
    'scipy',
    'wsgiref',
]

# Required before running setup()
SETUP_REQUIRES = [
    'setuptools>=18.0',
    'numpy',
    'cython',
]

# Dependencies to be fetched from urls (e.g. github repos)
LINKS_REQUIRES = []

# Dependencies to install for testing (e.g. nose or pytest)
TESTS_REQUIRES = []

# Dependencies to install for extra features
# For now, only documentation is enabled. Install with pip install -e .[doc]
EXTRA_REQUIRES = {
    'doc': ['sphinx'],
    # 'tests': TESTS_REQUIRES,
}

# Enable a handle to install all extra dependencies at once
# with pip install -e .[all]
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
