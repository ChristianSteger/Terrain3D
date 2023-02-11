# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import numba as nb


# -----------------------------------------------------------------------------

@nb.jit(nb.int64[:, :, :](nb.int64, nb.int64), nopython=True, parallel=True)
def get_quad_indices(num_ver_x, num_ver_y):
    """Compute quad indices for two-dimensional data.

    Parameters
    ----------
    num_ver_x : int
        Number of vertices in x-direction
    num_ver_y : int
        Number of vertices in y-direction

    Returns
    -------
    quad_indices : ndarray of double
        Array (three-dimensional; num_quad_y, num_quad_x, 5) with indices of
        quads' vertices"""

    num_quad_x = num_ver_x - 1
    num_quad_y = num_ver_y - 1
    quad_indices = np.empty((num_quad_y, num_quad_x, 5), dtype=np.int64)
    quad_indices[:, :, 0] = 4
    for i in nb.prange(num_quad_y):
        for j in range(num_quad_x):
            quad_indices[i, j, 1] = i * num_ver_x + j
            quad_indices[i, j, 2] = i * num_ver_x + j + 1
            quad_indices[i, j, 3] = (i + 1) * num_ver_x + j + 1
            quad_indices[i, j, 4] = (i + 1) * num_ver_x + j
    return quad_indices
