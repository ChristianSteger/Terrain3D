# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
import numba as nb


# -----------------------------------------------------------------------------

@nb.jit(nb.float64[:, :, :, :](nb.float64[:], nb.float64[:], nb.float32[:, :]),
        nopython=True)
def get_vertices(x_ver, y_ver, elevation_pad_0):
    """Compute vertices of grid cell columns.

    Parameters
    ----------
    x_ver : ndarray of double
        Array with x-coordinates of vertices
    y_ver : ndarray of double
        Array with y-coordinates of vertices
    elevation_pad_0 : ndarray of float
        Array with elevation data (array is padded with 0.0 at each four sides)

    Returns
    -------
    vertices : ndarray of double
        Array (four-dimensional; y, x, 4, vertex coordinates) with coordinates
        of vertices"""

    # Compute vertices
    num_ver_x = x_ver.size
    num_ver_y = y_ver.size
    vertices = np.empty((num_ver_y, num_ver_x, 4, 3), dtype=np.float64)
    for i in range(num_ver_y):
        for j in range(num_ver_x):
            vertices[i, j, 0, :] = [x_ver[j],
                                    y_ver[i],
                                    elevation_pad_0[i + 1, j + 1]]
            vertices[i, j, 1, :] = [x_ver[j],
                                    y_ver[i],
                                    elevation_pad_0[i + 1, j]]
            vertices[i, j, 2, :] = [x_ver[j],
                                    y_ver[i],
                                    elevation_pad_0[i, j]]
            vertices[i, j, 3, :] = [x_ver[j],
                                    y_ver[i],
                                    elevation_pad_0[i, j + 1]]
    return vertices


# -----------------------------------------------------------------------------

@nb.jit(nb.int64(nb.int64, nb.int64, nb.int64,
                 nb.types.containers.UniTuple(nb.types.int64, 4)),
        nopython=True)
def map_indices(ind_0, ind_1, ind_2, shp_ver):
    """Map indices for four-dimensional vertices array (y, x, 4, 3) to
    reshaped two-dimensional array (number of vertices, 3).

    Parameters
    ----------
    ind_0 : int
        First index for four-dimensional array
    ind_1 : int
        Second index for four-dimensional array
    ind_2 : int
        Third index for four-dimensional array
    shp_ver : tuple
        Dimension lengths of four-dimensional array

    Returns
    -------
    ind : int
        Index for two-dimensional array"""

    ind = ind_0 * shp_ver[1] * shp_ver[2] + ind_1 * shp_ver[2] + ind_2
    return ind


# -----------------------------------------------------------------------------

@nb.jit(nb.types.Tuple((nb.int64[:, :], nb.float32[:], nb.int64[:]))
        (nb.float32[:, :], nb.float32[:, :],
         nb.types.containers.UniTuple(nb.types.int64, 4)),
        nopython=True)
def get_quads(elevation, elevation_pad_0, shp_ver):
    """Compute quads of grid cell columns.

    Parameters
    ----------
    elevation : ndarray of float
        Array with elevation data
    elevation_pad_0 : ndarray of float
        Array with elevation data (array is padded with 0.0 at each four sides)
    shp_ver : tuple
        Dimension lengths of four-dimensional array

    Returns
    -------
    quads : ndarray of int
        Array (two-dimensional; number of quads, 5) with quads
    cell_data : ndarray of float
        Array (one-dimensional) with cell data
    column_index: ndarray of int
        Array (one-dimensional) with column indices. These values describe the
        affiliation of quads to a grid cell of the elevation data
        (with a linear index)"""

    shp_elev = elevation.shape
    num_quads_max = 5 * elevation.size - (shp_elev[0] - 1) * shp_elev[1] \
        - (shp_elev[1] - 1) * shp_elev[0]
    quads = np.empty((num_quads_max, 5), dtype=np.int64)
    cell_data = np.empty(num_quads_max, dtype=np.float32)
    column_index = np.empty(num_quads_max, dtype=np.int64)
    quads[:, 0] = 4
    ind = 0
    for i in range(shp_elev[0]):
        for j in range(shp_elev[1]):
            # ------------------- upper horizontal quad -----------------------
            quads[ind, 1:] = [map_indices(i, j, 0, shp_ver),
                              map_indices(i, j + 1, 1, shp_ver),
                              map_indices(i + 1, j + 1, 2, shp_ver),
                              map_indices(i + 1, j, 3, shp_ver)]
            cell_data[ind] = elevation[i, j]
            column_index[ind] = i * shp_elev[1] + j
            ind += 1
            # ------------------- side quad (positive y) ----------------------
            if elevation_pad_0[i + 1, j + 1] > elevation_pad_0[i + 2, j + 1]:
                quads[ind, 1:] = [map_indices(i + 1, j, 3, shp_ver),
                                  map_indices(i + 1, j + 1, 2, shp_ver),
                                  map_indices(i + 1, j + 1, 1, shp_ver),
                                  map_indices(i + 1, j, 0, shp_ver)]
                cell_data[ind] = elevation[i, j]
                column_index[ind] = i * shp_elev[1] + j
                ind += 1
            # ------------------- side quad (negative x) ----------------------
            if elevation_pad_0[i + 1, j + 1] > elevation_pad_0[i + 1, j]:
                quads[ind, 1:] = [map_indices(i, j, 0, shp_ver),
                                  map_indices(i + 1, j, 3, shp_ver),
                                  map_indices(i + 1, j, 2, shp_ver),
                                  map_indices(i, j, 1, shp_ver)]
                cell_data[ind] = elevation[i, j]
                column_index[ind] = i * shp_elev[1] + j
                ind += 1
            # ------------------- side quad (negative y) ----------------------
            if elevation_pad_0[i + 1, j + 1] > elevation_pad_0[i, j + 1]:
                quads[ind, 1:] = [map_indices(i, j + 1, 1, shp_ver),
                                  map_indices(i, j, 0, shp_ver),
                                  map_indices(i, j, 3, shp_ver),
                                  map_indices(i, j + 1, 2, shp_ver)]
                cell_data[ind] = elevation[i, j]
                column_index[ind] = i * shp_elev[1] + j
                ind += 1
            # ------------------- side quad (positive x) ----------------------
            if elevation_pad_0[i + 1, j + 1] > elevation_pad_0[i + 1, j + 2]:
                quads[ind, 1:] = [map_indices(i + 1, j + 1, 2, shp_ver),
                                  map_indices(i, j + 1, 1, shp_ver),
                                  map_indices(i, j + 1, 0, shp_ver),
                                  map_indices(i + 1, j + 1, 3, shp_ver)]
                cell_data[ind] = elevation[i, j]
                column_index[ind] = i * shp_elev[1] + j
                ind += 1
            # -----------------------------------------------------------------

    quads = quads[:ind, :]
    cell_data = cell_data[:ind]
    column_index = column_index[:ind]
    if quads.max() > ((shp_ver[0] * shp_ver[1] * shp_ver[2]) - 1):
        raise ValueError("Quad index out of bounds")

    return quads, cell_data, column_index


# -----------------------------------------------------------------------------

def add_frame_monochrome(depth_limit, elevation, x_ver, y_ver, vertices_rshp,
                         shp_ver):
    """Add monochrome frame to domain.

    Parameters
    ----------
    depth_limit : float
        Lower limit of frame
    elevation : ndarray of float
        Array with elevation data
    x_ver : ndarray of double
        Array with x-coordinates of vertices
    y_ver : ndarray of double
        Array with y-coordinates of vertices
    vertices_rshp : ndarray of double
        Array (two-dimensional; number of vertices, vertex coordinates)
        with coordinates of vertices
    shp_ver : tuple
        Dimension lengths of four-dimensional array

    Returns
    -------
    vertices_rshp : ndarray of double
        Updated array (two-dimensional; number of vertices, vertex coordinates)
        with coordinates of vertices
    quads_low : ndarray of int
        Array (two-dimensional; number of quads, 5) with frame quads"""

    # Add additional vertices
    num_quad_x = elevation.shape[1]
    num_quad_y = elevation.shape[0]
    vertices_add = np.empty((num_quad_x * 2 + num_quad_y * 2, 3),
                            dtype=np.float64)
    # ----------------------------- Lower border ------------------------------
    slic = slice(0, num_quad_x)
    vertices_add[slic, 0] = x_ver[:-1]
    vertices_add[slic, 1] = y_ver[0]
    # ----------------------------- Right border ------------------------------
    slic = slice(num_quad_x, num_quad_x + num_quad_y)
    vertices_add[slic, 0] = x_ver[-1]
    vertices_add[slic, 1] = y_ver[:-1]
    # ----------------------------- Upper border ------------------------------
    slic = slice(num_quad_x + num_quad_y, num_quad_x * 2 + num_quad_y)
    vertices_add[slic, 0] = x_ver[1:][::-1]
    vertices_add[slic, 1] = y_ver[-1]
    # ----------------------------- Left border -------------------------------
    slic = slice(num_quad_x * 2 + num_quad_y, num_quad_x * 2 + num_quad_y * 2)
    vertices_add[slic, 0] = x_ver[0]
    vertices_add[slic, 1] = y_ver[1:][::-1]
    vertices_add[:, 2] = depth_limit
    # -------------------------------------------------------------------------

    # Add additional quads
    ind_ini = vertices_rshp.shape[0]
    quads_low = np.empty((num_quad_x * 2 + num_quad_y * 2, 5), dtype=np.int64)
    quads_low[:, 0] = 4
    ind = 0
    # ----------------------------- Lower border ------------------------------
    for i in range(num_quad_x):
        quads_low[ind, 1:] = [map_indices(0, i + 1, 2, shp_ver),
                              map_indices(0, i, 2, shp_ver),
                              ind_ini + ind,
                              ind_ini + ind + 1]
        ind += 1
    # ----------------------------- Right border ------------------------------
    for i in range(num_quad_y):
        quads_low[ind, 1:] = [map_indices(i + 1, num_quad_x, 3, shp_ver),
                              map_indices(i, num_quad_x, 3, shp_ver),
                              ind_ini + ind,
                              ind_ini + ind + 1]
        ind += 1
    # ----------------------------- Upper border ------------------------------
    for i in range(num_quad_x - 1, -1, -1):
        quads_low[ind, 1:] = [map_indices(num_quad_y, i, 0, shp_ver),
                              map_indices(num_quad_y, i + 1, 0, shp_ver),
                              ind_ini + ind,
                              ind_ini + ind + 1]
        ind += 1
    # ----------------------------- Left border -------------------------------
    for i in range(num_quad_y - 1, 0, -1):
        quads_low[ind, 1:] = [map_indices(i, 0, 1, shp_ver),
                              map_indices(i + 1, 0, 1, shp_ver),
                              ind_ini + ind,
                              ind_ini + ind + 1]
        ind += 1
    quads_low[ind, 1:] = [map_indices(0, 0, 1, shp_ver),
                          map_indices(0 + 1, 0, 1, shp_ver),
                          ind_ini + ind,
                          ind_ini]
    ind += 1
    # -------------------------------------------------------------------------
    vertices_rshp = np.vstack((vertices_rshp, vertices_add))
    if quads_low.max() > (vertices_rshp.shape[0] - 1):
        raise ValueError("Quad index out of bounds")

    return vertices_rshp, quads_low


# -----------------------------------------------------------------------------

def add_frame_ocean(depth_limit, elevation, x_ver, y_ver, vertices_rshp,
                    shp_ver):
    """Add ocean frame to domain.

    Parameters
    ----------
    depth_limit : float
        Lower limit of frame
    elevation : ndarray of float
        Array with elevation data
    x_ver : ndarray of double
        Array with x-coordinates of vertices
    y_ver : ndarray of double
        Array with y-coordinates of vertices
    vertices_rshp : ndarray of double
        Array (two-dimensional; number of vertices, vertex coordinates)
        with coordinates of vertices
    shp_ver : tuple
        Dimension lengths of four-dimensional array

    Returns
    -------
    vertices_rshp : ndarray of double
        Updated array (two-dimensional; number of vertices, vertex coordinates)
        with coordinates of vertices
    quads_ocean : ndarray of int
        Array (two-dimensional; number of quads, 5) with frame quads (ocean)
    cell_data_ocean : ndarray of float
        Array (one-dimensional) with cell data
    quads_low : ndarray of int
        Array (two-dimensional; number of quads, 5) with frame quads (lower)"""

    # Check arguments
    num_quads_ocean = ((elevation[0, :] < 0.0).sum()
                       + (elevation[-1, :] < 0.0).sum()
                       + (elevation[:, 0] < 0.0).sum()
                       + (elevation[:, -1] < 0.0).sum())
    if num_quads_ocean == 0:
        raise ValueError("Border domain of 'elevation' does not contain "
                         + "ocean grid cell(s). Use 'monochrome' frame.")
    if depth_limit > elevation.min():
        print("Warning: 'depth_limit' must be equal or smaller than "
              + "minimal elevation. Reset to %.1f" % elevation.min())
        depth_limit = elevation.min()

    # Add 'ocean' vertices
    num_quad_x = elevation.shape[1]
    num_quad_y = elevation.shape[0]
    vertices_ocean = np.empty((num_quads_ocean * 2, 3), dtype=np.float64)
    ind = 0
    # ----------------------------- Lower border ------------------------------
    for i in range(num_quad_x):
        if elevation[0, i] < 0.0:
            vertices_ocean[ind, :] = [x_ver[i],
                                      y_ver[0],
                                      elevation[0, i]]
            ind += 1
            vertices_ocean[ind, :] = [x_ver[i + 1],
                                      y_ver[0],
                                      elevation[0, i]]
            ind += 1
    # ----------------------------- Right border ------------------------------
    for i in range(num_quad_y):
        if elevation[i, -1] < 0.0:
            vertices_ocean[ind, :] = [x_ver[-1],
                                      y_ver[i],
                                      elevation[i, -1]]
            ind += 1
            vertices_ocean[ind, :] = [x_ver[-1],
                                      y_ver[i + 1],
                                      elevation[i, -1]]
            ind += 1
    # ----------------------------- Upper border ------------------------------
    for i in range(num_quad_x, 0, -1):
        if elevation[-1, i - 1] < 0.0:
            vertices_ocean[ind, :] = [x_ver[i],
                                      y_ver[-1],
                                      elevation[-1, i - 1]]
            ind += 1
            vertices_ocean[ind, :] = [x_ver[i - 1],
                                      y_ver[-1],
                                      elevation[-1, i - 1]]
            ind += 1
    # ----------------------------- Left border -------------------------------
    for i in range(num_quad_y, 0, -1):
        if elevation[i - 1, 0] < 0.0:
            vertices_ocean[ind, :] = [x_ver[0],
                                      y_ver[i],
                                      elevation[i - 1, 0]]
            ind += 1
            vertices_ocean[ind, :] = [x_ver[0],
                                      y_ver[i - 1],
                                      elevation[i - 1, 0]]
            ind += 1
    # -------------------------------------------------------------------------

    # Add 'lower' vertices
    num_quads_low = num_quad_x * 2 + num_quad_y * 2
    vertices_low = np.empty((num_quads_low, 3),
                            dtype=np.float64)
    # ----------------------------- Lower border ------------------------------
    slic = slice(0, num_quad_x)
    vertices_low[slic, 0] = x_ver[:-1]
    vertices_low[slic, 1] = y_ver[0]
    # ----------------------------- Right border ------------------------------
    slic = slice(num_quad_x, num_quad_x + num_quad_y)
    vertices_low[slic, 0] = x_ver[-1]
    vertices_low[slic, 1] = y_ver[:-1]
    # ----------------------------- Upper border ------------------------------
    slic = slice(num_quad_x + num_quad_y, num_quad_x * 2 + num_quad_y)
    vertices_low[slic, 0] = x_ver[1:][::-1]
    vertices_low[slic, 1] = y_ver[-1]
    # ----------------------------- Left border -------------------------------
    slic = slice(num_quad_x * 2 + num_quad_y, num_quad_x * 2 + num_quad_y * 2)
    vertices_low[slic, 0] = x_ver[0]
    vertices_low[slic, 1] = y_ver[1:][::-1]
    vertices_low[:, 2] = depth_limit
    # -------------------------------------------------------------------------

    # Add 'ocean' quads
    ind_ini = vertices_rshp.shape[0]
    quads_ocean = np.empty((num_quads_ocean, 5), dtype=np.int64)
    cell_data_ocean = np.empty(num_quads_ocean, dtype=np.float32)
    quads_ocean[:, 0] = 4
    ind = 0
    # ----------------------------- Lower border ------------------------------
    for i in range(num_quad_x):
        if elevation[0, i] < 0.0:
            quads_ocean[ind, 1:] = [map_indices(0, i + 1, 2, shp_ver),
                                    map_indices(0, i, 2, shp_ver),
                                    ind_ini + (ind * 2),
                                    ind_ini + (ind * 2) + 1]
            cell_data_ocean[ind] = elevation[0, i]
            ind += 1
    # ----------------------------- Right border ------------------------------
    for i in range(num_quad_y):
        if elevation[i, -1] < 0.0:
            quads_ocean[ind, 1:] = [map_indices(i + 1, num_quad_x, 3, shp_ver),
                                    map_indices(i, num_quad_x, 3, shp_ver),
                                    ind_ini + (ind * 2),
                                    ind_ini + (ind * 2) + 1]
            cell_data_ocean[ind] = elevation[i, -1]
            ind += 1
    # ----------------------------- Upper border ------------------------------
    for i in range(num_quad_x - 1, -1, -1):
        if elevation[-1, i] < 0.0:
            quads_ocean[ind, 1:] = [map_indices(num_quad_y, i, 0, shp_ver),
                                    map_indices(num_quad_y, i + 1, 0, shp_ver),
                                    ind_ini + (ind * 2),
                                    ind_ini + (ind * 2) + 1]
            cell_data_ocean[ind] = elevation[-1, i]
            ind += 1
    # ----------------------------- Left border -------------------------------
    for i in range(num_quad_y - 1, -1, -1):
        if elevation[i, 0] < 0.0:
            quads_ocean[ind, 1:] = [map_indices(i, 0, 1, shp_ver),
                                    map_indices(i + 1, 0, 1, shp_ver),
                                    ind_ini + (ind * 2),
                                    ind_ini + (ind * 2) + 1]
            cell_data_ocean[ind] = elevation[i, 0]
            ind += 1
    # -------------------------------------------------------------------------
    vertices_rshp = np.vstack((vertices_rshp, vertices_ocean, vertices_low))
    if quads_ocean.max() > (vertices_rshp.shape[0] - 1):
        raise ValueError("Ocean quad index out of bounds")

    # Add 'lower' quads
    ind_low_ini = vertices_rshp.shape[0] - vertices_low.shape[0]
    ind_ocean = ind_low_ini - vertices_ocean.shape[0]
    quads_low = np.empty((num_quads_low, 5), dtype=np.int64)
    quads_low[:, 0] = 4
    ind = 0
    # ----------------------------- Lower border ------------------------------
    for i in range(num_quad_x):
        if elevation[0, i] < 0.0:
            quads_low[ind, 1:] = [ind_ocean + 1,
                                  ind_ocean,
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
            ind_ocean += 2
        else:
            quads_low[ind, 1:] = [map_indices(0, i + 1, 2, shp_ver),
                                  map_indices(0, i, 2, shp_ver),
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
        ind += 1
    # ----------------------------- Right border ------------------------------
    for i in range(num_quad_y):
        if elevation[i, -1] < 0.0:
            quads_low[ind, 1:] = [ind_ocean + 1,
                                  ind_ocean,
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
            ind_ocean += 2
        else:
            quads_low[ind, 1:] = [map_indices(i + 1, num_quad_x, 3, shp_ver),
                                  map_indices(i, num_quad_x, 3, shp_ver),
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
        ind += 1
    # ----------------------------- Upper border ------------------------------
    for i in range(num_quad_x - 1, -1, -1):
        if elevation[-1, i] < 0.0:
            quads_low[ind, 1:] = [ind_ocean + 1,
                                  ind_ocean,
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
            ind_ocean += 2
        else:
            quads_low[ind, 1:] = [map_indices(num_quad_y, i, 0, shp_ver),
                                  map_indices(num_quad_y, i + 1, 0, shp_ver),
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
        ind += 1
    # ----------------------------- Left border -------------------------------
    for i in range(num_quad_y - 1, 0, -1):
        if elevation[i, 0] < 0.0:
            quads_low[ind, 1:] = [ind_ocean + 1,
                                  ind_ocean,
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
            ind_ocean += 2
        else:
            quads_low[ind, 1:] = [map_indices(i, 0, 1, shp_ver),
                                  map_indices(i + 1, 0, 1, shp_ver),
                                  ind_low_ini + ind,
                                  ind_low_ini + ind + 1]
        ind += 1
    if elevation[0, 0] < 0.0:
        quads_low[ind, 1:] = [ind_ocean + 1,
                              ind_ocean,
                              ind_low_ini + ind,
                              ind_low_ini]
        ind_ocean += 2
    else:
        quads_low[ind, 1:] = [map_indices(0, 0, 1, shp_ver),
                              map_indices(0 + 1, 0, 1, shp_ver),
                              ind_low_ini + ind,
                              ind_low_ini]
    ind += 1
    # -------------------------------------------------------------------------
    if quads_low.max() > (vertices_rshp.shape[0] - 1):
        raise ValueError("Low quad index out of bounds")

    return vertices_rshp, quads_ocean, cell_data_ocean, quads_low
