# Description: Visualise entire GEBCO data set on sphere with a triangle mesh.
#              The elevation of quad vertices, which are below sea level and
#              are land according to the GSHHG data base, are set to 0.0 m.
#              Ice covered quads (land glaciers or ice shelves) are represented
#              as white areas.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import os
import numpy as np
import vtk
import pyvista as pv
from cmcrameri import cm
from pyproj import CRS
from pyproj import Transformer
from scipy import interpolate
import time
from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# mpl.style.use("classic")

# Load required functions
sys.path.append("/Users/csteger/Downloads/Terrain3D/functions/")
from gebco import get as get_gebco
from outlines import binary_mask as binary_mask_outlines
from triangles import get_quad_indices

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# res_ter = 0.1  # resolution of visualised terrain [degree]
# gebco_agg_num = 24  # aggregation number of input GEBCO data [-]
res_ter = 0.05  # resolution of visualised terrain [degree]
gebco_agg_num = 12  # aggregation number of input GEBCO data [-]

ter_exa_fac = 25.0  # terrain exaggeration factor [-]

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Check input settings
if not (180 / res_ter).is_integer():
    raise ValueError("Selected input value for 'res_ter' is invalid")

# Create quad vertices (longitude: [-180.0, +180.0], latitude [-90.0, +90.0])
num_ver_lat = int(180 / res_ter) + 1
num_ver_lon = int(360 / res_ter) + 1
lon_ver = np.linspace(-180.0, 180.0, num_ver_lon, dtype=np.float64)
lat_ver = np.linspace(-90.0, 90.0, num_ver_lat, dtype=np.float64)
lon_ver_2d, lat_ver_2d = np.meshgrid(lon_ver, lat_ver)
print("Dimensions of interpolated GEBCO data: "
      + str(lat_ver.size) + " x " + str(lon_ver.size))

# Get GEBCO data
lon_in, lat_in, elevation_in, crs_in = get_gebco(gebco_agg_num)
print("Dimensions of input GEBCO data: "
      + str(lat_in.size) + " x " + str(lon_in.size))

# Interpolate GEBCO data to quad vertices (bi-linearly)
t_beg = time.time()
f_ip_rbs = interpolate.RectBivariateSpline(lat_in, lon_in, elevation_in,
                                           kx=1, ky=1)
elevation_ver = f_ip_rbs(lat_ver, lon_ver).astype(np.float32)
# -> values outside the domain are filled via nearest-neighbor extrapolation
print("GEBCO data interpolated (%.1f" % (time.time() - t_beg) + " s)")

# Set elevation of quad vertices, which are land and below sea level, to 0.0 m
mask_land = binary_mask_outlines("shorelines", lon_ver, lat_ver, crs_in,
                                 resolution="intermediate", level=1,
                                 sub_sample_num=1)
mask_lakes = binary_mask_outlines("shorelines", lon_ver, lat_ver, crs_in,
                                  resolution="intermediate", level=2,
                                  sub_sample_num=1)
mask_land[mask_lakes] = False
mask_lbsl = (elevation_ver < 0.0) & mask_land  # mask with land below sea level
elevation_ver[mask_lbsl] = 0.0

# # Plot mask with land below sea level
# plt.figure()
# plt.pcolormesh(lon_ver, lat_ver, mask_lbsl)
# plt.colorbar()

# Ensure that 'cyclic boundary values' are identical
dev = np.abs(np.array([elevation_ver[0, :].min(), elevation_ver[0, :].max()])
             - elevation_ver[0, :].mean()).max()
elevation_ver[0, :] = elevation_ver[0, :].mean()
print("Max. abs. elevation deviation (South Pole): %.1f" % dev + " m")
dev = np.abs(np.array([elevation_ver[-1, :].min(), elevation_ver[-1, :].max()])
             - elevation_ver[-1, :].mean()).max()
elevation_ver[-1, :] = elevation_ver[-1, :].mean()
print("Max. abs. elevation deviation (North Pole): %.1f" % dev + " m")
elevation_mean = (elevation_ver[:, 0] + elevation_ver[:, -1]) / 2.0
dev = np.abs(np.append(elevation_ver[:, 0] - elevation_mean,
                       elevation_ver[:, -1] - elevation_mean)).max()
elevation_ver[:, 0] = elevation_mean
elevation_ver[:, -1] = elevation_mean
print("Max. abs. elevation deviation (+/-180.0 deg): %.1f" % dev + " m")

# Convert geographic latitudes/longitudes to cartesian coordinates
t_beg = time.time()
crs_sphere = CRS.from_proj4("+proj=latlon +ellps=sphere")
crs_cart = CRS.from_proj4("+proj=geocent +ellps=sphere")
transformer = Transformer.from_crs(crs_sphere, crs_cart)
x, y, z = transformer.transform(lon_ver_2d, lat_ver_2d,
                                elevation_ver * ter_exa_fac)
print("Convert geographic to cartesian coordinates "
      + "(%.1f" % (time.time() - t_beg) + " s)")

# Create indices array for quad vertices
t_beg = time.time()
num_quad_lon = num_ver_lon - 1
num_quad_lat = num_ver_lat - 1
quad_indices = get_quad_indices(num_ver_lon, num_ver_lat)
print("Create quad vertices array (%.1f" % (time.time() - t_beg) + " s)")

# Create mask for glaciated area (for quads)
lon_quad = lon_ver[:-1] + np.diff(lon_ver) / 2.0
lat_quad = lat_ver[:-1] + np.diff(lat_ver) / 2.0
mask_ice = np.zeros((num_quad_lat, num_quad_lon), dtype=bool)
mask_glacier_land = binary_mask_outlines("glacier_land", lon_quad, lat_quad,
                                         crs_in, sub_sample_num=1)
mask_ice[mask_glacier_land] = True
mask_ice_shelves = binary_mask_outlines("antarctic_ice_shelves", lon_quad,
                                        lat_quad, crs_in, sub_sample_num=1)
mask_ice[mask_ice_shelves] = True
del mask_glacier_land, mask_ice_shelves

# # Plot ice mask
# plt.figure()
# plt.pcolormesh(lon_quad, lat_quad, mask_ice)
# plt.colorbar()

# Reshape arrays
vertices = np.hstack((x.ravel()[:, np.newaxis],
                      y.ravel()[:, np.newaxis],
                      z.ravel()[:, np.newaxis]))
quad_indices = quad_indices.reshape(num_quad_lat * num_quad_lon, 5)
if quad_indices.max() >= vertices.shape[0]:
    raise ValueError("Index out of bounds!")

# Create mesh for terrain
mask_ice_rav = mask_ice.ravel()
quad_sel = quad_indices[~mask_ice_rav, :]
cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
cell_types[:] = vtk.VTK_QUAD
grid = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)
grid.point_data["Surface elevation"] = elevation_ver.ravel()
# -> point data related to vertices -> color values are interpolated across
#    edges (https://docs.pyvista.org/user-guide/what-is-a-mesh.html)

# Create mesh for ice-covered area
quad_sel = quad_indices[mask_ice_rav, :]
cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
cell_types[:] = vtk.VTK_QUAD
grid_ice = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)

# -----------------------------------------------------------------------------
# Visualise data
# -----------------------------------------------------------------------------

# Colormap
num_cols = 256
mapping = np.linspace(elevation_ver.min(), elevation_ver.max(), num_cols)
cols = np.empty((num_cols, 4), dtype=np.float32)
for i in range(num_cols):
    if mapping[i] < 0.0:
        val = (1.0 - mapping[i] / mapping[0]) / 2.0
        cols[i, :] = cm.bukavu(val)
    else:
        val = (mapping[i] / mapping[-1]) / 2.0 + 0.5
        cols[i, :] = cm.bukavu(val)
colormap = ListedColormap(cols)

# Plot
pl = pv.Plotter(window_size=[1000, 1000])
col_bar_args = dict(height=0.25, vertical=True, position_x=0.8, position_y=0.1)
pl.add_mesh(grid, scalars="Surface elevation", show_edges=False, label="1",
            edge_color="black", line_width=0, cmap=colormap,
            scalar_bar_args=col_bar_args)
pl.add_mesh(grid_ice, color="white", show_edges=False, label="1",
            edge_color="black", line_width=0)
pl.set_background("black")
pl.remove_scalar_bar()
pl.show()
