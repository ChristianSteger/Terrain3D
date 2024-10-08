# Description: Visualise entire GEBCO data set on sphere with a triangle mesh.
#              The elevation of quad vertices, which are below sea level and
#              are land according to the GSHHG data base, are set to 0.0 m.
#              Ice covered quads (land glaciers or ice shelves) are represented
#              as white areas.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import terrain3d

mpl.style.use("classic") # type: ignore

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# res_ter = 0.1  # resolution of visualised terrain [degree]
# gebco_agg_num = 24  # aggregation number of input GEBCO data [-]
res_ter = 0.05  # resolution of visualised terrain [degree]
gebco_agg_num = 12  # aggregation number of input GEBCO data [-]
ter_exa_fac = 25.0  # terrain exaggeration factor [-]
path_out = os.getenv("HOME") + "/Desktop/PyVista_globe/" # type: ignore
# output directory for animation

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
lon_in, lat_in, elevation_in, crs_in = terrain3d.gebco.get(gebco_agg_num)
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
mask_land = terrain3d.outlines.binary_mask(
    "shorelines", lon_ver, lat_ver, crs_in, sub_sample_num=1,
    resolution="intermediate", level=1)
mask_lakes = terrain3d.outlines.binary_mask(
    "shorelines", lon_ver, lat_ver, crs_in, sub_sample_num=1,
    resolution="intermediate", level=2)
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
quad_indices = terrain3d.tri_mesh.get_quad_indices(num_ver_lon, num_ver_lat)
print("Create quad vertices array (%.1f" % (time.time() - t_beg) + " s)")

# Create mask for glaciated area (for quads)
lon_quad = lon_ver[:-1] + np.diff(lon_ver) / 2.0
lat_quad = lat_ver[:-1] + np.diff(lat_ver) / 2.0
mask_ice = np.zeros((num_quad_lat, num_quad_lon), dtype=bool)
mask_glacier_land = terrain3d.outlines.binary_mask(
    "glacier_land", lon_quad, lat_quad, crs_in, sub_sample_num=1)
mask_ice[mask_glacier_land] = True
mask_ice_shelves = terrain3d.outlines.binary_mask(
    "antarctic_ice_shelves", lon_quad, lat_quad, crs_in, sub_sample_num=1)
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
cmap = terrain3d.auxiliary.terrain_colormap(elevation_ver)

# Plot
pl = pv.Plotter(window_size=[2000, 2000])
col_bar_args = dict(height=0.25, vertical=True, position_x=0.8, position_y=0.1)
pl.add_mesh(grid, scalars="Surface elevation", show_edges=False, cmap=cmap,
            scalar_bar_args=col_bar_args)
pl.add_mesh(grid_ice, color="white", show_edges=False)
pl.set_background("black") # type: ignore
pl.remove_scalar_bar() # type: ignore
pl.show()

# -----------------------------------------------------------------------------
# Create animation (--> orbiting around globe)
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise ValueError("Output directory does not exist")

# Compute camera orbit
num_frames = 500  # number of frames (for movie)
azimuth = np.linspace(0.0, 360.0 - (360 / num_frames), num_frames)
elevation = np.repeat(0.0, num_frames)

# # Assume that azimuth and elevation are valid for rotated coordinates
# geo_crs = ccrs.PlateCarree()
# rot_pole_crs = crs_rot = CRS.from_user_input(
#     ccrs.RotatedPole(pole_latitude=(90.0 - 47.4),
#                      pole_longitude=-171.5,
#                      central_rotated_longitude=0.0)
# )
# coord_geo = geo_crs.transform_points(rot_pole_crs, azimuth, elevation)[:, :2]
# azimuth, elevation = coord_geo[:, 0], coord_geo[:, 1]

# # Check orbit
# plt.figure(figsize=(12, 5))
# plt.scatter(azimuth, elevation)
# plt.xlabel("Geographic longitude [deg]")
# plt.ylabel("Geographic latitude [deg]")

# Create images
pl = pv.Plotter(window_size=[1500, 1500], off_screen=True)
pl.add_mesh(grid, scalars="Surface elevation", show_edges=False, cmap=cmap,
            show_scalar_bar=False)
pl.add_mesh(grid_ice, color="white", show_edges=False)
pl.set_background("black") # type: ignore
pl.camera_position = "yz"
pl.camera.zoom(1.5)
for i in range(num_frames):
    pl.camera.azimuth = azimuth[i]
    pl.camera.elevation = elevation[i]
    pl.render()
    pl.screenshot(path_out + "fig_" + "%03d" % (i + 1) + ".png")
pl.close()

# Create movie from images (requires 'FFmpeg')
# ffmpeg -framerate 60 -i fig_%03d.png -c:v libx264 -pix_fmt yuv420p globe.mp4
