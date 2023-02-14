# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import zipfile
import numpy as np
import xarray as xr
import time
import numba as nb
from pyproj import CRS


# -----------------------------------------------------------------------------

def _download(path_data_root):
    """Download gridded GEBCO bathymetry and terrain data.

    Source of data:
    https://www.gebco.net/data_and_products/gridded_bathymetry_data/

    Parameters
    ----------
    path_data_root : str
        Local root path for downloaded data"""

    # Check arguments
    if not os.path.isdir(path_data_root):
        raise ValueError("Local root path does not exist")

    # Download data
    file_url = "https://www.bodc.ac.uk/data/open_download/gebco/" \
               + "gebco_2022/zip/"
    path_gebco = path_data_root + "gebco"
    if not os.path.isdir(path_gebco):
        print(" Download GEBCO data ".center(79, "-"))
        print("~7.5 GB of storage required - proceed with download "
              + "in specified directory (yes/no)?")
        cont = ""
        flag = False
        while cont not in ("yes", "no"):
            if flag:
                print("Please enter 'yes' or 'no'")
            cont = input()
            flag = True
        if cont == "yes":
            file_zipped = path_gebco + ".zip"
            download_file(file_url, file_zipped)
            with zipfile.ZipFile(file_zipped, "r") as zip_ref:
                zip_ref.extractall(path_gebco)
            os.remove(file_zipped)


# -----------------------------------------------------------------------------

@nb.jit(nb.float32[:, :](nb.float32[:, :], nb.int64), nopython=True,
        parallel=True)
def _aggregate_slice(data, agg_num):
    """Aggregate data along first two dimensions by computing mean.

    Parameters
    ----------
    data : ndarray of float
        Array (two-dimensional) with input data
    agg_num : int
        Spatial aggregation number

    Returns
    -------
    data_agg : ndarray of float
        Array (two-dimensional) with aggregated data"""

    shp_0 = data.shape[0] // agg_num
    shp_1 = data.shape[1] // agg_num
    data_agg = np.empty((shp_0, shp_1), dtype=np.float32)
    for j in nb.prange(shp_0):
        for k in range(shp_1):
            slice_0 = slice(j * agg_num, (j + 1) * agg_num)
            slice_1 = slice(k * agg_num, (k + 1) * agg_num)
            data_agg[j, k] = data[slice_0, slice_1].mean()
    return data_agg


# -----------------------------------------------------------------------------

def _aggregate_spatially(file_gebco, agg_num):
    """Spatially aggregate gridded GEBCO bathymetry and terrain data.

    Parameters
    ----------
    file_gebco : str
        Path and file name of GEBCO NetCDF file
    agg_num : int
        Spatial aggregation number"""

    # Check arguments
    if not os.path.isfile(file_gebco):
        raise ValueError("GEBCO NetCDF not found in specified location")
    if (agg_num < 1) or (agg_num > 100):
        raise ValueError("Aggregation number out of range [1, 100]")
    ds = xr.open_dataset(file_gebco)
    len_lon = ds.coords["lon"].size
    len_lat = ds.coords["lat"].size
    ds.close()
    agg_num_all = np.arange(1, 101, 1)
    mask_valid = (len_lon % agg_num_all == 0) \
        & ((len_lat // 4) % agg_num_all == 0)
    if agg_num not in agg_num_all[mask_valid]:
        ind = np.where(agg_num_all == agg_num)[0][0]
        agg_num_valid = (agg_num_all[np.where(mask_valid[:ind])[0][-1]],
                         agg_num_all[ind + np.where(mask_valid[ind:])[0][0]])
        print(agg_num_valid)
        raise ValueError("Invalid aggregation number - closest valid "
                         + "numbers are: " + str(agg_num_valid[0])
                         + ", " + str(agg_num_valid[1]))

    # Compute aggregation number 1 (-> use symbolic link)
    if not os.path.isfile(file_gebco[:-3] + "_agg_num_001.nc"):
        print(" Spatially aggregate GEBCO data (aggregation number: 1) "
              .center(79, "-"))
        os.symlink(file_gebco, file_gebco[:-3] + "_agg_num_001.nc")

    # Compute remaining aggregation numbers (-> compute in four steps due to
    # large size of input data)
    if not os.path.isfile(file_gebco[:-3] + "_agg_num_"
                          + str(agg_num).zfill(3) + ".nc"):
        print((" Spatially aggregate GEBCO data (aggregation number: "
               + str(agg_num) + ") ").center(79, "-"))

        t_beg = time.time()

        # Output arrays
        shp_out = (len_lat // agg_num, len_lon // agg_num)
        elevation_agg = np.empty(shp_out, dtype=np.float32)
        lat_agg = np.empty(shp_out[0], dtype=np.float64)
        lon_agg = np.empty(shp_out[1], dtype=np.float64)

        # Loop through latitudinal slices
        for i in range(4):

            # Load data
            ds = xr.open_dataset(file_gebco)
            ds = ds.isel(lat=slice(i * (len_lat // 4),
                                   (i + 1) * (len_lat // 4)))
            lat = ds["lat"].values
            lon = ds["lon"].values
            elevation = ds["elevation"].values.astype(np.float32)
            ds.close()

            # Spatially aggregate data
            slice_agg = slice(i * (lat_agg.size // 4),
                              (i + 1) * (lat_agg.size // 4))
            elevation_agg[slice_agg, :] \
                = _aggregate_slice(elevation, agg_num).astype(np.int16)
            lat_agg[slice_agg] = np.mean(lat.reshape(int(lat.size / agg_num),
                                                     agg_num), axis=1)
            if i == 0:
                lon_agg[:] = np.mean(lon.reshape(int(lon.size / agg_num),
                                                 agg_num), axis=1)
            del lon, lat

            print("Completed step: " + str(i + 1) + "/4")

        # Save aggregated data to NetCDF file
        ds = xr.open_dataset(file_gebco)
        ds = ds.isel(lat=slice(0, lat_agg.size), lon=slice(0, lon_agg.size))
        ds = ds.assign_coords(lat=lat_agg)
        ds = ds.assign_coords(lon=lon_agg)
        ds["elevation"].values = elevation_agg.astype(np.int16)
        ds.to_netcdf(file_gebco[:-3] + "_agg_num_" + str(agg_num).zfill(3)
                     + ".nc")

        print("Processing time: %.1f" % (time.time() - t_beg) + " s")


# -----------------------------------------------------------------------------

def get(agg_num, domain=(-180.0, 180.0, -90.0, 90.0)):
    """Get gridded GEBCO bathymetry and terrain data (spatially aggregated)
    for a specific geographic domain.

    Parameters
    ----------
    agg_num : int
        Spatial aggregation number
    domain : tuple, optional
        Boundaries of geographic domain (lon_min, lon_max, lat_min, lat_max)
        [degree]

    Returns
    -------
    lon_gebco : ndarray of double
        Array (two-dimensional) with geographic longitude [degree]
    lat_gebco : ndarray of double
        Array (two-dimensional) with geographic latitude [degree]
    elevation_gebco : ndarray of float
        Array (two-dimensional) with elevation [metre]
    crs_gebco : pyproj.crs.crs.CRS
        Geospatial reference system of GEBCO"""

    # Check arguments
    if (domain[0] < -180.0) or (domain[1] > 180.0):
        raise ValueError("Longitude range out of bounds")
    if (domain[2] < -90.0) or (domain[3] > 90.0):
        raise ValueError("Latitude range out of bounds")

    # Ensure that GEBCO data was downloaded
    path_data_root = "/Users/csteger/Dropbox/IAC/Temp/Terrain_3D/data/"
    _download(path_data_root)

    # Ensure that GEBCO data with aggregation number exists
    file_gebco = path_data_root + "gebco/GEBCO_2022.nc"
    _aggregate_spatially(file_gebco, agg_num)

    # Load spatially aggregated GEBCO data for domain
    ds = xr.open_dataset(file_gebco[:-3] + "_agg_num_" + str(agg_num).zfill(3)
                         + ".nc")
    ds = ds.sel(lat=slice(domain[2], domain[3]),
                lon=slice(domain[0], domain[1]))
    lat_gebco = ds["lat"].values  # float64, [degree]
    lon_gebco = ds["lon"].values  # float64, [degree]
    elevation_gebco = ds["elevation"].values  # int16, [metre]
    crs_gebco = CRS.from_string(ds["crs"].epsg_code)
    ds.close()

    return lon_gebco, lat_gebco, elevation_gebco, crs_gebco
