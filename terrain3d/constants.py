# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np

# -----------------------------------------------------------------------------
# Definition of constants
# -----------------------------------------------------------------------------

radius_earth = 6370997.0  # Earth radius [m]
deg2m = (2.0 * np.pi * radius_earth) / 360.0
# equatorial distance per degree [m deg-1]
