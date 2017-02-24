#!/usr/bin/env python
# -*- coding: utf-8 -*-

PACKAGE_NAME = 'phantomas'

def main():
    """ Install entry-point """
    from os import path as op
    from glob import glob
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from io import open  # pylint: disable=W0622
    import numpy as np

    this_path = op.dirname(op.abspath(getfile(currentframe())))

    # Python 3: use a locals dictionary
    # http://stackoverflow.com/a/1463370/6820620
    ldict = locals()

    # Get version and release info, which is all stored in phantomas/info.py
    module_file = op.join(this_path, PACKAGE_NAME, 'info.py')
    with open(module_file) as infofile:
        pythoncode = [line for line in infofile.readlines() if not line.strip().startswith('#')]
        exec('\n'.join(pythoncode), globals(), ldict)


    extensions = [Extension(
            "phantomas.mr_simul.fast_volume_fraction",
            ["phantomas/mr_simul/fast_volume_fraction.pyx",
            "phantomas/mr_simul/c_fast_volume_fraction.c"],
            include_dirs=[np.get_include(), "/usr/local/include/"],
            library_dirs=["/usr/lib/"],
            libraries=["gsl", "gslcblas"]),
    ]

    setup(
        name=PACKAGE_NAME,
        version=ldict['__version__'],
        description=ldict['__description__'],
        long_description=ldict['__longdesc__'],
        author=ldict['__author__'],
        author_email=ldict['__email__'],
        maintainer=ldict['__maintainer__'],
        maintainer_email=ldict['__email__'],
        license=ldict['__license__'],
        url=ldict['URL'],
        download_url=ldict['DOWNLOAD_URL'],
        classifiers=ldict['CLASSIFIERS'],
        packages=find_packages(exclude=['build', 'doc', 'examples', 'scripts']),
        package_data={'phantomas.mr_simul': ["spherical_21_design.txt"]},
        scripts=glob('scripts/*'),
        ext_modules=extensions,
        zip_safe=False,
        # Dependencies handling
        setup_requires=ldict['SETUP_REQUIRES'],
        install_requires=ldict['REQUIRES'],
        dependency_links=ldict['LINKS_REQUIRES'],
        tests_require=ldict['TESTS_REQUIRES'],
        extras_require=ldict['EXTRA_REQUIRES'],
    )


if __name__ == '__main__':
    main()

