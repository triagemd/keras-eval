#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-eval',
    version='0.0.34',
    description='An evaluation abstraction for Keras models.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        'numpy==1.14.5',
        'tensorflow==1.10',
        'Keras',
        'h5py',
        'Pillow',
        'scipy',
        'numpy',
        'pandas',
        'plotly',
        'matplotlib',
        'sklearn',
        'keras-model-specs>=0.0.28',
        'ipykernel==5.0.0',
        'setuptools<=39.1.0'
    ]
)
