# Description: Setup file
#
# Installation of package: python -m pip install .
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
from setuptools import setup

setup(
    name="terrain3d",
    version="0.1",
    description="Three-dimensional visualisation of terrain data from digital"
                "elevation models (DEMs) or climate model topography with"
                "PyVista. A set of Python example scripts illustrates how this"
                "data can be plotted for various domains and with auxiliary"
                "information.",
    author="Christian R. Steger",
    author_email="christian.steger@env.ethz.ch",
    packages=["terrain3d"]
)
