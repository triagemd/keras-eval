#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-eval',
    version='0.0.40',
    description='An evaluation abstraction for Keras models.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        'Keras>=2.2.4',
        'numpy',
        'h5py',
        'Pillow',
        'scipy',
        'pandas',
        'plotly',
        'matplotlib',
        'sklearn',
        'keras-model-specs>=0.0.30',
        'setuptools==39.1.0'
    ]
)
