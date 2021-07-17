###
# Copyright (c) 2021-present, Kirill Chirkunov
# Licensed under The MIT License [see LICENSE for details]
###

import os

import setuptools
from setuptools import setup, find_packages

def find_recursive_packages(root):
    def _isdir(f):
        return os.path.isdir(f) and '__pycache__' not in f

    dirs = list(filter(_isdir, [os.path.join(root, f) for f in os.listdir(root)]))
    for dir_ in dirs:
        dirs += find_recursive_packages(dir_)
    
    return [dir_.replace('/', '.') for dir_ in dirs]
    
NAME = 'neurospatio'
VERSION = 0.4
URL = 'https://github.ibm.com/lliryc'
DESCRIPTION = 'A Python module based on the Keras/Tensorflow to spread spatial properties over the surface'
LONG_DESCRIPTION = None
AUTHOR = 'Kirill Chirkunov'
AUTHOR_EMAIL = 'kirill.chirkunov@gmail.com'
KEYWORDS = 'spatial, interpolation, ANN, keras, tensorflow'
REQUIRES_PYTHON = '>=3.8'

try:
    import pypandoc
    LONG_DESCRIPTION = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    keywords=KEYWORDS,
    python_requires=REQUIRES_PYTHON,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Geoscience/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ]
)