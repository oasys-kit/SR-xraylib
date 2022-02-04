#! /usr/bin/env python3

import imp
import os
import subprocess

try:
    from setuptools import find_packages, setup
except AttributeError:
    from setuptools import find_packages, setup

NAME = 'srxraylib'

VERSION = '1.0.33'
ISRELEASED = True

DESCRIPTION = 'Synchrotron Radiation X-ray library'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Luca Rebuffi, Manuel Sanchez del Rio and Mark Glass'
AUTHOR_EMAIL = 'lrebuffi@anl.gov'
URL = 'https://github.com/oasys-kit/SR-xraylib'
DOWNLOAD_URL = 'https://github.com/oasys-kit/SR-xraylib'
MAINTAINER = 'Luca Rebuffi'
MAINTAINER_EMAIL = 'lrebuffi@anl.gov'
LICENSE = 'GPLv3'

KEYWORDS = (
    'x-ray'
    'synchrotron radiation',
    'wavefront propagation'
    'ray tracing',
    'surface metrology',
    'simulation',
)

CLASSIFIERS = (
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: '
    'GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
)

INSTALL_REQUIRES = (
    'setuptools',
    'numpy',
    'scipy',
)

SETUP_REQUIRES = (
    'setuptools',
)

PACKAGES = [
    "srxraylib",
    "srxraylib.metrology",
    "srxraylib.sources",
    "srxraylib.util",
    "srxraylib.waveoptics",
    "srxraylib.plot",
]

PACKAGE_DATA = {
    "srxraylib.metrology": ["*.txt"],
    "srxraylib.sources": ["data/*.*"],
    "srxraylib.util": ["data/*.*"],
    "srxraylib.optics": ["data/*.*"],
}


def setup_package():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        # extra setuptools args
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        setup_requires=SETUP_REQUIRES,
    )

if __name__ == '__main__':
    setup_package()
