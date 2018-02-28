#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-eval',
    version='0.0.1',
    description='A evaluation abstraction for Keras models.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        'Keras',
        'h5py',
        'tensorflow',
        'Pillow',
        'scipy',
        'numpy',
        'matplotlib',
        'sklearn',
        'keras-model-specs>=0.0.16',
    ]
)
