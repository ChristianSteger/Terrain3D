# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import zipfile
import numpy as np
import time
import fiona
from shapely.geometry import shape
from shapely.ops import transform as transform_shapely
from rasterio.transform import Affine
from rasterio.features import rasterize
from pyproj import CRS
from pyproj import Transformer

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from descartes import PolygonPatch
# mpl.style.use("classic")

# Load required functions
sys.path.append("/Users/csteger/Downloads/Terrain3D/functions/")
from auxiliary import download_file


# -----------------------------------------------------------------------------

def _download(path_data_root, product):
    """Download different shapefiles with outlines (shorelines, glaciated land
    area or Antarctic ice shelves)

    Parameters
    ----------
    path_data_root : str
        Local root path for downloaded data
    product : str
        Product to download ('shorelines', 'glacier_land' or
        'antarctic_ice_shelves')"""

    # Check arguments
    if not os.path.isdir(path_data_root):
        raise ValueError("Local root path does not exist")
    if product not in ("shorelines", "glacier_land",
                       "antarctic_ice_shelves"):
        raise ValueError("Unknown product. Known products are 'shorelines',"
                         + " 'glacier_land' or 'antarctic_ice_shelves'")

    # Product information
    product_info = \
        {"shorelines":
            {"url": "http://www.soest.hawaii.edu/pwessel/gshhg/"
                    + "gshhg-shp-2.3.7.zip",
             "folder": "gshhg",
             "print": "GSHHG global shoreline database"},
         "glacier_land":
             {"url": "https://www.naturalearthdata.com/http//"
                     + "www.naturalearthdata.com/download/10m/"
                     + "physical/ne_10m_glaciated_areas.zip",
              "folder": "ne_10m_glacier_land",
              "print": "glaciated area outlines from Natural Earth"},
         "antarctic_ice_shelves":
             {"url": "https://www.naturalearthdata.com/http//"
                     + "www.naturalearthdata.com/download/10m/"
                     + "physical/ne_10m_antarctic_ice_shelves_polys.zip",
              "folder": "ne_10m_antarctic_ice_shelves",
              "print": "antarctic ice shelf outlines from Natural Earth"}}

    # Download data
    path_data = path_data_root + product_info[product]["folder"]
    if not os.path.isdir(path_data):
        print((" Download " + product_info[product]["print"] + " ")
              .center(79, "-"))
        file_zipped = path_data + ".zip"
        download_file(product_info[product]["url"], file_zipped)
        with zipfile.ZipFile(file_zipped, "r") as zip_ref:
            zip_ref.extractall(path_data)
        os.remove(file_zipped)


# -----------------------------------------------------------------------------

def binary_mask(product, x, y, crs_grid, resolution="intermediate",
                level=1):
    """Compute binary mask from outlines.

    Parameters
    ----------
    product : str
        Outline product ('shorelines', 'glacier_land' or
        'antarctic_ice_shelves')
    x : ndarray of double
        Array (1-dimensional) with x-coordinates [arbitrary]
    y : ndarray of double
        Array (1-dimensional) with y-coordinates [arbitrary]
    crs_grid : pyproj.crs.crs.CRS
        Geospatial reference system of gridded input data
    resolution : str
        Resolution of shoreline outlines (either 'crude', 'low',
        'intermediate', 'high' or 'full')
    level : 1
        Level of shoreline data ('1' to '6'). Available levels are:
        1: Continental land masses and ocean islands, except Antarctica
        2: Lakes
        3: Islands in lakes
        4: Ponds in islands within lakes
        5: Antarctica based on ice front boundary
        6: Antarctica based on grounding line boundary

    Returns
    -------
    mask_bin : ndarray of bool
        Binary mask (locations inside outlines equal True)"""

    # Check arguments
    if product not in ("shorelines", "glacier_land",
                       "antarctic_ice_shelves"):
        raise ValueError("Unknown product. Known products are 'shorelines',"
                         + " 'glacier_land' or 'antarctic_ice_shelves'")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x-coordinates are not strictly increasing")
    if not np.all(np.diff(y) > 0):
        raise ValueError("y-coordinates values not strictly increasing")
    if resolution not in ("crude", "low", "intermediate", "high", "full"):
        raise ValueError("Invalid value for 'resolution'. Valid values are "
                         + "'crude', 'low', 'intermediate', 'high' or 'full'")
    if (level < 1) or (level > 6):
        raise ValueError("Value for 'level' out of range [1 - 6]")

    # Ensure that required data was downloaded
    path_data_root = "/Users/csteger/Dropbox/IAC/Temp/Terrain_3D/data/"
    _download(path_data_root, product)

    print((" Compute binary mask for " + product + " outlines ")
          .center(79, "-"))

    # Determine shapefile
    shp_prod = {
        "shorelines":
            path_data_root + "gshhg/GSHHS_shp/" + resolution[0]
            + "/GSHHS_" + resolution[0] + "_L" + str(level) + ".shp",
        "glacier_land":
            path_data_root + "ne_10m_glacier_land/ne_10m_glaciated_areas.shp",
        "antarctic_ice_shelves":
            path_data_root + "ne_10m_antarctic_ice_shelves/"
            + "ne_10m_antarctic_ice_shelves_polys.shp"
    }

    # Load polygons (large polygons are split)
    ds = fiona.open(shp_prod[product])
    crs_outlines = CRS.from_string(ds.crs["init"])
    polygons = [shape(i["geometry"]) for i in ds]
    ds.close()

    # Transform polygons in case geospatial reference systems are different
    if crs_grid == crs_outlines:
        print("Both data sets share the same geospatial reference system "
              + "(EPSG:" + str(crs_grid.to_epsg()) + ")")
    else:
        print("Transform polygons to geospatial reference system of grid data")
        t_beg = time.time()
        project = Transformer.from_crs(crs_outlines, crs_grid,
                                       always_xy=True).transform
        polygons_trans = []
        for i in polygons:
            polygons_trans.append(transform_shapely(project, i))
        polygons = polygons_trans
        print("Processing time: %.1f" % (time.time() - t_beg) + " s")

    # Compute binary mask
    print("Compute binary mask")
    t_beg = time.time()
    dx = np.diff(x).mean()
    dy = np.diff(y).mean()
    transform = Affine(dx, 0.0, x[0] - dx / 2.0,
                       0.0, dy, y[0] - dy / 2.0)
    mask_bin = rasterize(polygons, (len(y), len(x)), transform=transform)
    print("Processing time: %.1f" % (time.time() - t_beg) + " s")

    # # Test plot
    # plt.figure()
    # ax = plt.axes()
    # plt.pcolormesh(x, y, mask_bin)
    # for i in polygons[:1000]:
    #     poly = PolygonPatch(i, facecolor="none", edgecolor="yellow")
    #     ax.add_patch(poly)

    return mask_bin.astype(bool)
