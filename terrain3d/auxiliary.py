# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from tqdm import tqdm
import requests
import numpy as np
from pyproj import CRS, Transformer
from matplotlib.colors import ListedColormap


# -----------------------------------------------------------------------------

def download_file(file_url, file_path_local, auth=None):
    """Download single file from web and show progress with bar.

    Parameters
    ----------
    file_url : str
        URL of file to download
    file_path_local : str
        Local path for downloaded file
    auth : tuple
        Tuple (username, password) for enabling HTTP authentication """

    # Check arguments
    if not os.path.isdir(os.path.dirname(file_path_local)):
        raise ValueError("Directory for local file does not exist")
    if auth is not None:
        if ((not isinstance(auth, tuple)) or (len(auth) != 2)
                or any([not isinstance(i, str) for i in auth])):
            raise ValueError("'auth' must be tuple with 2 strings")

    # Try to download file
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)"
                          "" + "AppleWebKit/537.36 (KHTML, like Gecko) "
                          + "Chrome/39.0.2171.95 Safari/537.36"}
        response = requests.get(file_url, stream=True, headers=headers,
                                auth=auth)
    except requests.exceptions.SSLError:
        print("SSL certificate verification failed - continue download "
              + "(yes/no)?")
        cont = ""
        flag = False
        while cont not in ("yes", "no"):
            if flag:
                print("Please enter 'yes' or 'no'")
            cont = input()
            flag = True
        if cont == "yes":
            response = requests.get(file_url, stream=True, verify=False)
        else:
            return
    if response.ok:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 10
        # download seems to be faster with larger block size...
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                            unit_scale=True)
        with open(file_path_local, "wb") as infile:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                infile.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ValueError("Inconsistency in file size")
    else:
        raise ValueError("Response not ok (status code: "
                         + str(response.status_code) + ")")


# -----------------------------------------------------------------------------

def aggregate_dem(x, y, elevation, agg_num):
    """Spatially aggregate digital elevation model (DEM) data. The dimensions
    of the DEM are cropped in case they can not evenly be divided by the
    aggregation number.

    Parameters
    ----------
    x : ndarray of float/double [arbitrary]
        Array with x-coordinates of digital elevation model
    y : ndarray of float/double [arbitrary]
        Array with y-coordinates of digital elevation model
    elevation : ndarray of float/double
        Array with elevation of DEM [arbitrary]
    agg_num : int
        Spatial aggregation number

    Returns
    -------
    quad_indices : ndarray of double
        Array (3-dimensional; num_quad_y, num_quad_x, 5) with indices of
        quads' vertices"""

    # Check arguments
    if elevation.dtype not in (np.float32, np.float64):
        raise TypeError("Array 'elevation' must be of type 'np.float32' "
                        + "or 'np.float64)'")

    # Crop DEM if necessary
    x_len_valid = (x.size // agg_num) * agg_num
    y_len_valid = (y.size // agg_num) * agg_num
    if (x.size != x_len_valid) or (y.size != y_len_valid):
        print("Warning: dimensions are cropped from " + str(elevation.shape)
              + " to (" + str(y_len_valid) + ", " + str(x_len_valid) + ")")
        x = x[:x_len_valid]
        y = y[:y_len_valid]
        elevation = elevation[:y_len_valid, :x_len_valid]

    if (x.size % agg_num != 0) or (y.size % agg_num != 0):
        raise ValueError("Invalid aggregation number")

    # Aggregate data
    x_agg = np.mean(x.reshape(int(x.size / agg_num), agg_num), axis=1)
    y_agg = np.mean(y.reshape(int(y.size / agg_num), agg_num), axis=1)

    y = np.arange(0, elevation.shape[0], agg_num)
    temp = np.add.reduceat(elevation, y, axis=0, dtype=elevation.dtype)
    x = np.arange(0, elevation.shape[1], agg_num)
    elevation_agg = np.add.reduceat(temp, x, axis=1, dtype=elevation.dtype)
    elevation_agg /= float(agg_num * agg_num)

    return x_agg, y_agg, elevation_agg


# -----------------------------------------------------------------------------

def gridcoord(x_cent, y_cent):
    """Compute edge coordinates from grid cell centre coordinates.

    Parameters
    ----------
    x_cent: array_like
        Array (one-dimensional) with x-coordinates of grid centres [arbitrary]
    y_cent: array_like
        Array (one-dimensional) with y-coordinates of grid centres [arbitrary]

    Returns
    -------
    x_edge: array_like
        Array (one-dimensional) with x-coordinates of grid edges [arbitrary]
    y_edge: array_like
        Array (one-dimensional) with y-coordinates of grid edges [arbitrary]"""

    # Check input arguments
    if len(x_cent.shape) != 1 or len(y_cent.shape) != 1:
        raise TypeError("number of dimensions of input arrays is not 1")
    if (np.any(np.diff(np.sign(np.diff(x_cent))) != 0) or
            np.any(np.diff(np.sign(np.diff(y_cent))) != 0)):
        sys.exit("input arrays are not monotonically in- or decreasing")

    # Compute grid spacing if not provided
    dx = np.diff(x_cent).mean()
    dy = np.diff(y_cent).mean()

    # Compute grid coordinates
    x_edge = np.hstack((x_cent[0] - (dx / 2.),
                        x_cent[:-1] + np.diff(x_cent) / 2.,
                        x_cent[-1] + (dx / 2.))).astype(x_cent.dtype)
    y_edge = np.hstack((y_cent[0] - (dy / 2.),
                        y_cent[:-1] + np.diff(y_cent) / 2.,
                        y_cent[-1] + (dy / 2.))).astype(y_cent.dtype)

    return x_edge, y_edge


# -----------------------------------------------------------------------------

def domain_extend_geo_coord(x_cent, y_cent, crs_proj, bound_res,
                            domain_ext=0.1):
    """Compute a domain in geographic coordinates (WGS84) required to
    conservatively remap from this domain to the map projection domain.

    Parameters
    ----------
    x_cent : ndarray of float/double [degree or m]
        Array with x-coordinates of map projection
    y_cent : ndarray of float/double [degree or m]
        Array with y-coordinates of map projection
    crs_proj : pyproj.crs.crs.CRS
        Geospatial reference of map projection
    bound_res : float
        Resolution of boundary [degree or m]
    domain_ext : float
        Domain extension ('safety' margin for remapping) [degree]

    Returns
    -------
    domain : tuple
        Boundaries of domain (lon_min, lon_max, lat_min, lat_max) in WGS84
        [degree]"""

    # Check arguments
    if (not np.all(np.diff(x_cent) > 0.0)) \
            or (not np.all(np.diff(y_cent) > 0.0)):
        sys.exit("Both input coordinates array must contain monotonically "
                 + "increasing values")
    if bound_res > np.minimum(np.diff(x_cent).mean(), np.diff(y_cent).mean()):
        raise ValueError("'bound_res' must be equal or smaller than minimal"
                         + " resolution of 'x_cent' and 'y_cent'")

    # Compute required domain
    x_lim = (x_cent[0] - np.diff(x_cent).mean() / 2.0,
             x_cent[-1] + np.diff(x_cent).mean() / 2.0)
    y_lim = (y_cent[0] - np.diff(y_cent).mean() / 2.0,
             y_cent[-1] + np.diff(y_cent).mean() / 2.0)
    x_edge = np.linspace(x_lim[0], x_lim[1],
                         int(np.ceil((x_lim[1] - x_lim[0]) / bound_res)))
    y_edge = np.linspace(y_lim[0], y_lim[1],
                         int(np.ceil((y_lim[1] - y_lim[0]) / bound_res)))
    x_bound = np.hstack((x_edge,
                         np.repeat(x_edge[-1], len(y_edge))[1:],
                         x_edge[::-1][1:],
                         np.repeat(x_edge[0], len(y_edge))[1:]))
    y_bound = np.hstack((np.repeat(y_edge[0], len(x_edge)),
                         y_edge[1:],
                         np.repeat(y_edge[-1], len(x_edge))[1:],
                         y_edge[::-1][1:]))
    transformer = Transformer.from_crs(crs_proj, CRS.from_epsg(4326),
                                       always_xy=True)
    lon_bound, lat_bound = transformer.transform(x_bound, y_bound)
    domain = [lon_bound.min() - domain_ext,
              lon_bound.max() + domain_ext,
              lat_bound.min() - domain_ext,
              lat_bound.max() + domain_ext]

    return domain


# -----------------------------------------------------------------------------

def cmap_terrain(elevation, cmap_in, num_cols=256):
    """Adapt terrain colormap to make it suitable for PyVista.

    Parameters
    ----------
    elevation : ndarray of float/double
        Array with elevation of DEM [arbitrary]
    cmap_in : matplotlib.colors.ListedColormap
        Colormap suitable for terrain representation (lower half ocean and
        upper half land terrain; like 'cmcrameri.cm.bukavu')
    num_cols : int
        Number of colours

    Returns
    -------
    cmap_out : matplotlib.colors.ListedColormap
        Colormap adapted for use in PyVista"""

    mapping = np.linspace(elevation.min(), elevation.max(), num_cols)
    cols = np.empty((num_cols, 4), dtype=np.float32)
    for i in range(num_cols):
        if mapping[i] < 0.0:
            val = (1.0 - mapping[i] / mapping[0]) / 2.0
            cols[i, :] = cmap_in(val)
        else:
            val = (mapping[i] / mapping[-1]) / 2.0 + 0.5
            cols[i, :] = cmap_in(val)
    cmap_out = ListedColormap(cols)

    return cmap_out