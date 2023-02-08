# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import tarfile
import numpy as np
import time
import rasterio
import glob
from rasterio.merge import merge

# Load required functions
sys.path.append("/Users/csteger/Downloads/Terrain3D/functions/")
from auxiliary import download_file


# -----------------------------------------------------------------------------

def _download_tile(path_data_root, tile):
    """Download MERIT digital elevation model tile (30 x 30 degree) in GeoTiff
    format.

    Source of data:
    http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/index.html

    Parameters
    ----------
    path_data_root : str
        Local root path for downloaded data
    tile : str
        Tile name (e.g. 'dem_tif_n30e000')"""

    # Check arguments
    if not os.path.isdir(path_data_root):
        raise ValueError("Local root path does not exist")

    # Download data
    path_merit = path_data_root + "merit/"
    if not os.path.isdir(path_merit):
        os.makedirs(path_merit)
    if not os.path.isdir(path_merit + tile):
        print("A username and password is required to download MERIT data.")
        print("Visit the following webpage for registration:")
        print("http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/index.html")
        print("Enter username:")
        username = input()
        print("Enter password:")
        password = input()
        file_url = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/" \
                   + "distribute/v1.0.2/" + tile + ".tar"
        file_tar = path_merit + tile + ".tar"
        download_file(file_url, file_tar, auth=(username, password))
        tar = tarfile.open(file_tar)
        tar.extractall(path_merit)
        tar.close()
        os.remove(file_tar)
    else:
        print("Tile " + tile + " already downloaded")


# -----------------------------------------------------------------------------

def get(domain):
    """Get MERIT digital elevation model data for a specific geographic domain.

    Parameters
    ----------
    domain : tuple, optional
        Boundaries of geographic domain (lon_min, lon_max, lat_min, lat_max)
        [degree]

    Returns
    -------
    lon_merit : ndarray of double
        Array (two-dimensional) with geographic longitude [degree]
    lat_merit : ndarray of double
        Array (two-dimensional) with geographic latitude [degree]
    elevation_merit : ndarray of float
        Array (two-dimensional) with elevation [metre]"""

    # Check arguments
    if (domain[0] < -180.0) or (domain[1] > 180.0):
        raise ValueError("Longitude range out of bounds")
    if (domain[2] < -60.0) or (domain[3] > 90.0):
        raise ValueError("Latitude range out of bounds")

    # Missing tiles (-> ocean)
    tiles_ocean = ("dem_tif_n00w150", "dem_tif_s60w150", "dem_tif_s60w120")

    # Determine required 30 x 30 degree domain
    domain_30x30 = (int(np.floor(domain[0] / 30) * 30),
                    int(np.ceil(domain[1] / 30) * 30),
                    int(np.floor(domain[2] / 30) * 30),
                    int(np.ceil(domain[3] / 30) * 30))

    # Ensure that required MERIT tile(s) were/was downloaded
    path_data_root = "/Users/csteger/Dropbox/IAC/Temp/Terrain_3D/data/"
    lon_pre = ("w", "e", "e")
    lat_pre = ("s", "n", "n")
    for i in range(domain_30x30[0], domain_30x30[1], 30):
        for j in range(domain_30x30[2], domain_30x30[3], 30):
            tile = "dem_tif_" + lat_pre[np.sign(j) + 1] + str(j).zfill(2) \
                   + lon_pre[np.sign(i) + 1] + str(i).zfill(3)
            if tile not in tiles_ocean:
                _download_tile(path_data_root, tile)
            else:
                raise ValueError("Domain overlaps with missing MERIT tile "
                                 + "(ocean) -> this case is not yet "
                                 + "implemented")

    # Determine required 5 x 5 degree domain
    domain_5x5 = (int(np.floor(domain[0] / 5) * 5),
                  int(np.ceil(domain[1] / 5) * 5),
                  int(np.floor(domain[2] / 5) * 5),
                  int(np.ceil(domain[3] / 5) * 5))

    # Merge and load relevant GeoTiff sub-tiles
    path_merit = path_data_root + "merit/"
    print(" Merge and load GeoTiff sub-tiles ".center(79, "-"))
    t_beg = time.time()
    sub_tiles = []
    for i in range(domain_5x5[0], domain_5x5[1], 5):
        for j in range(domain_5x5[2], domain_5x5[3], 5):
            sub_tile = path_merit + "*/" \
                       + lat_pre[np.sign(j) + 1] + str(j).zfill(2) \
                       + lon_pre[np.sign(i) + 1] + str(i).zfill(3) + "_dem.tif"
            sub_tiles.append(glob.glob(sub_tile)[0])
    mosaic, out_trans = merge(sub_tiles)
    print("Processing time: %.1f" % (time.time() - t_beg) + " s")

    # Set ocean values to 0.0 m
    mosaic[mosaic == -9999.0] = 0.0

    # Create coordinates
    d_lon = out_trans[0]
    d_lat = out_trans[4]
    lon_ulc = out_trans[2]
    lat_ulc = out_trans[5]
    raster_size_x = mosaic.shape[2]
    raster_size_y = mosaic.shape[1]
    lon_edge = np.linspace(lon_ulc, lon_ulc + d_lon * raster_size_x,
                           raster_size_x + 1)
    lat_edge = np.linspace(lat_ulc, lat_ulc + d_lat * raster_size_y,
                           raster_size_y + 1)
    lon = lon_edge[:-1] + np.diff(lon_edge / 2.0)
    lat = lat_edge[:-1] + np.diff(lat_edge / 2.0)

    # Crop relevant domain
    slice_lon = slice(np.argmin(np.abs(domain[0] - lon)),
                      np.argmin(np.abs(domain[1] - lon)) + 1)
    slice_lat = slice(np.argmin(np.abs(domain[3] - lat)),
                      np.argmin(np.abs(domain[2] - lat)) + 1)
    elevation_merit = mosaic[0, slice_lat, slice_lon].astype(np.float32)
    lon_merit, lat_merit = lon[slice_lon], lat[slice_lat]
    del mosaic, out_trans

    # Reverse latitude coordinates (-> change order to ascending)
    lat_merit = lat_merit[::-1]
    elevation_merit = np.flipud(elevation_merit)

    return lon_merit, lat_merit, elevation_merit
