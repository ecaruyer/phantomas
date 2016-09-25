"""
Phantomas
=========

*Phantomas* is an open-source software library for the creation of *realistic
phantoms* in *diffusion MRI*. This is intented as a tool for the quantitative
evaluation of methods in acquisition, signal and image processing, local
reconstruction and fiber tracking in diffusion MRI. Phantomas is released
under the terms of the revised BSD license. You should have received a copy
of the license together with the software, please refer to the file
``LICENSE``.

"""
from __future__ import absolute_import
from .info import (
    __version__,
    __author__,
    __email__,
    __maintainer__,
    __copyright__,
    __credits__,
    __license__,
    __status__,
    __description__,
    __longdesc__
)

__all__ = ['geometry', 'mr_simul', 'utils', 'visu']

