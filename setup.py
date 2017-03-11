#!/usr/bin/env python
from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

version = sys.version_info

# numpy support for python3.3 not available for version > 1.10.1
if version.major == 3 and version.minor == 3:
    NUMPY_VERSION = 'numpy >= 1.9.2, <= 1.10.1'
else:
    NUMPY_VERSION = 'numpy >= 1.9.2'


class PyTest(TestCommand, object):

    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        super(PyTest, self).initialize_options()
        self.pytest_args = []

    def finalize_options(self):
        super(PyTest, self).finalize_options()
        self.test_suite = True
        self.test_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        exit(pytest.main(self.pytest_args))


readme = open('README.md').read()
doclink = """
Documentation
-------------

The full documentation is at <url to be inserted>."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='emf-rbm',
    version='0.1.0',
    description='Some description',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Charles Martin',
    author_email='basaks@gmail.com',
    url='https://github.com/charlesmartin14/emf-rbm',
    packages=['.'],
    package_dir={'emf-rbm': 'emfrbm'},
    include_package_data=True,
    entry_points={
        'console_scripts': []
    },
    setup_requires=[NUMPY_VERSION],
    install_requires=[
        'numpy >= 1.9.2',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'h5py',
        'nltk'
    ],
    extras_require={
        'dev': [
            'sphinx',
            'ghp-import',
            'sphinxcontrib-programoutput'
        ]
    },
    tests_require=[
        'pytest-cov',
        'coverage',
        'codecov',
        'tox',
        'pytest'  # pytest should be last
    ],
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='RBM, Deep Learning, Restricted Boltzmann Machine',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        # "Programming Language :: Python :: 3.7",
        # add additional supported python versions
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis"
        # add more topics
    ],
    cmdclass={
        'test': PyTest,
    }
)
