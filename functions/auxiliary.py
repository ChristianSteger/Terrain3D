# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from tqdm import tqdm
import requests
import numpy as np
import numba as nb


# -----------------------------------------------------------------------------

def download_file(file_url, file_path_local, auth=None):
    """Download single file from web and show progress with bar.

    Parameters
    ----------
    file_url : str
        URL of file to download
    file_path_local: str
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
