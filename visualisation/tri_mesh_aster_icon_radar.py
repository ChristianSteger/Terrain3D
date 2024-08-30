# Description: Raw ASTER 30 m resolution DEM vs. ICON 1km resolution 
#              topography with radar location
#
# Copyright (c) 2024 MeteoSwiss, Christian R. Steger

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
from pyproj import CRS
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib as mpl
import terrain3d
import xarray as xr
from cmcrameri import cm

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Miscellaneous
ter_exa_fac = 2.0  # terrain exaggeration factor [-]
show_lakes = True  # represent lakes as blue areas
# path_examp = "/home/stc/Desktop/data/"
# path_examp = "/Users/csteger/Dropbox/MeteoSwiss/Terrain3D/"
path_examp = terrain3d.auxiliary.get_path_data() + "example_data/"

# Radar location (longitude, latitude, altitude) [deg, deg, m asl]
radar_loc = (8.833217, 46.040791, 1626)  # Monte Lema
# radar_loc = (7.486552, 46.370646, 2937)  # Pointe de la Plaine Morte
# radar_loc = (8.512000, 47.284333, 938)  # Albis
# radar_loc = (6.099415, 46.425113, 1682)  # La Dole
# radar_loc = (9.794458, 46.834974, 2850)  # Weissfluhgipfel

# Domain width [km]
domain_width = 22.5

# Radar visualisation
radar_radius = 200
# radar_colour = "dodgerblue"
# radar_colour = "#EC5800"
radar_colour = "black"

# -----------------------------------------------------------------------------
# Preparation
# -----------------------------------------------------------------------------

# Domain extent
domain_width_lat = ((domain_width * 1000.0) / terrain3d.constants.deg2m)
domain_width_lon = domain_width_lat * (1.0 / np.cos(np.deg2rad(radar_loc[1])))
fac_lat = 1.0 / np.cos(np.deg2rad(radar_loc[1]))
domain = (radar_loc[0] - domain_width_lon / 2.0,
          radar_loc[0] + domain_width_lon / 2.0,
          radar_loc[1] - domain_width_lat / 2.0,
          radar_loc[1] + domain_width_lat / 2.0)

# Download example data
path_examp = terrain3d.auxiliary.get_path_data() + "example_data/"
if not os.path.exists(path_examp):
    os.makedirs(path_examp)
for i in ["ASTER_orig_T031.nc",
          "icon_grid_0001_R19B08_mch_DOM01.nc",
          "lfff00000000c.nc"]:
    terrain3d.auxiliary.download_file(
        "https://github.com/ChristianSteger/Example_data/blob/main/Terrain3D/"
        + i + "?raw=true", path_examp + i)

# -----------------------------------------------------------------------------
# Prepare DEM data (ASTER)
# -----------------------------------------------------------------------------

# Load data
ds = xr.open_dataset(path_examp + "ASTER_orig_T031.nc")
ds = ds.sel(lon=slice(domain[0], domain[1]), lat=slice(domain[3], domain[2]))
lon_ver = ds["lon"].values
# Reverse latitude coordinates (-> change order to ascending)
lat_ver = ds["lat"].values[::-1]
elevation_ver = np.flipud(ds["Z"].values)
ds.close()
crs_dem = CRS.from_string("EPSG:4326")
# -------------------
# ds = xr.open_dataset(path_data + "ASTER_orig_T031.nc")
# ds_sel = ds.sel(lon=slice(7.0, 9.25), lat=slice(46.75, 45.5))
# comp = dict(zlib=True, complevel=1)
# encoding = {var: comp for var in ds_sel.data_vars}
# ds_sel.to_netcdf(path_data + "var_sel/ASTER_orig_T031.nc", encoding=encoding)
# -------------------

# Compute binary lake mask (optional)
if show_lakes:
    lon_quad = lon_ver[:-1] + np.diff(lon_ver) / 2.0
    lat_quad = lat_ver[:-1] + np.diff(lat_ver) / 2.0
    mask_lake = terrain3d.outlines.binary_mask(
        "swiss_lakes", lon_quad, lat_quad, crs_dem, sub_sample_num=5)

# plt.figure()
# plt.pcolormesh(lon_ver, lat_ver, mask_lake, shading="auto")
# plt.colorbar()
# plt.show()

# Transform geographic coordinates to orthographic projection
crs_wgs84 = CRS.from_string("EPSG:4326")
crs_ortho = CRS.from_dict({"proj": "ortho", "lat_0": lat_ver.mean(),
                           "lon_0": lon_ver.mean(), "ellps": "WGS84"})
transformer = Transformer.from_crs(crs_wgs84, crs_ortho, always_xy=True)
x, y, z = transformer.transform(*np.meshgrid(lon_ver, lat_ver),
                                elevation_ver)  # type: ignore

# # Test plot (2D)
# cmap_mpt = terrain3d.auxiliary.truncate_colormap(cm.bukavu, (0.5, 1.0))
# levels = np.arange(0.0, 4000.0, 250.0)
# norm_mpt = mpl.colors.BoundaryNorm(levels, ncolors=cmap_mpt.N, extend="both")
# plt.figure()
# plt.pcolormesh(x, y, z, shading="auto", norm=norm_mpt, cmap=cmap_mpt)
# plt.colorbar()
# plt.show()

# Create indices array for quad vertices
num_quad_x = len(lon_ver) - 1
num_quad_y = len(lat_ver) - 1
quad_indices = terrain3d.tri_mesh.get_quad_indices(len(lon_ver), len(lat_ver))

# Reshape arrays
vertices = np.hstack((x.ravel()[:, np.newaxis],
                      y.ravel()[:, np.newaxis],
                      z.ravel()[:, np.newaxis] * ter_exa_fac))
quad_indices = quad_indices.reshape(num_quad_y * num_quad_x, 5)
if quad_indices.max() >= vertices.shape[0]:
    raise ValueError("Index out of bounds!")

# Create mesh
if not show_lakes:
    cell_types = np.empty(quad_indices.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_dem = pv.UnstructuredGrid(quad_indices.ravel(), cell_types, vertices)
    grid_dem.point_data["Surface elevation"] = z.ravel()
else:
    mask_lake_rav = mask_lake.ravel()
    quad_sel = quad_indices[~mask_lake_rav, :]
    cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_dem = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)
    grid_dem.point_data["Surface elevation"] = z.ravel()

# Create mesh for lake-covered area (optional)
if show_lakes:
    quad_sel = quad_indices[mask_lake_rav, :]
    cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_dem_lake = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)

# -----------------------------------------------------------------------------
# Prepare ICON data
# -----------------------------------------------------------------------------

# Load data
ds = xr.open_dataset(path_examp + "icon_grid_0001_R19B08_mch_DOM01.nc")
clon = np.rad2deg(ds["clon"].values)
clat = np.rad2deg(ds["clat"].values)
vlon = np.rad2deg(ds["vlon"].values)
vlat = np.rad2deg(ds["vlat"].values)
vertex_of_cell = ds["vertex_of_cell"].values - 1
cells_of_vertex = ds["cells_of_vertex"].values - 1
ds.close()
# -------------------
# ds_sel = ds[["clon", "clat", "vlon", "vlat", "vertex_of_cell", "cells_of_vertex"]]
# comp = dict(zlib=True, complevel=1)
# encoding = {var: comp for var in ds_sel.data_vars}
# ds_sel.to_netcdf(path_data + "var_sel/icon_grid_0001_R19B08_mch_DOM01.nc", encoding=encoding)
# -------------------

ds = xr.open_dataset(path_examp + "lfff00000000c.nc")
hsurf = ds["HSURF"].values
mask_water = (ds["soiltyp"].values == 9)
ds.close()
# -------------------
# ds_sel = ds[["HSURF", "soiltyp"]].drop_vars(["clon", "clat"])
# comp = dict(zlib=True, complevel=1)
# encoding = {var: comp for var in ds_sel.data_vars}
# ds_sel.to_netcdf(path_data + "var_sel/lfff00000000c.nc", encoding=encoding)
# -------------------

# Compute elevation at vertices as average of adjacent cells
hsurf_vertex = np.empty(vlon.size, dtype=np.float64)
for i in range(vlon.size):
    ind = cells_of_vertex[:, i]
    hsurf_vertex[i] = hsurf[ind[ind >= 0]].mean()

# Transform geographic coordinates to orthographic projection
cx, cy, cz = transformer.transform(clon, clat, hsurf)
vx, vy, vz = transformer.transform(vlon, vlat, hsurf_vertex)

# Mask with relevant cells
mask_sel = (vlon[vertex_of_cell].min(axis=0) >= domain[0]) \
    & (vlon[vertex_of_cell].max(axis=0) <= domain[1]) \
    & (vlat[vertex_of_cell].min(axis=0) >= domain[2]) \
    & (vlat[vertex_of_cell].max(axis=0) <= domain[3])

# # Test plot (2D)
# triangles = mpl.tri.Triangulation(vx, vy, vertex_of_cell.transpose())
# plt.figure()
# plt.tripcolor(vx, vy, triangles.triangles[mask_sel, :], cz[mask_sel],
#               cmap=cmap_mpt, norm=norm_mpt)
# plt.colorbar()
# plt.show()

# Create mesh
if not show_lakes:
    mask = mask_sel
else:
    mask = mask_sel & ~mask_water
vertices = np.concatenate((vx[:, np.newaxis],
                           vy[:, np.newaxis],
                           vz[:, np.newaxis] * ter_exa_fac), axis=1)
tri_indices = np.hstack([np.full(mask.sum(), 3)[:, np.newaxis],
                            vertex_of_cell[:, mask].transpose()])
cell_types = np.empty(mask.sum(), dtype=np.uint8)
cell_types[:] = vtk.VTK_TRIANGLE
grid_icon = pv.UnstructuredGrid(tri_indices.ravel(), cell_types, vertices)
grid_icon.cell_data["Surface elevation"] = cz[mask]

# Create mesh for lake-covered area (optional)
if show_lakes:
    mask = mask_sel & mask_water
    vertices = np.concatenate((vx[:, np.newaxis],
                            vy[:, np.newaxis],
                            vz[:, np.newaxis] * ter_exa_fac), axis=1)
    tri_indices = np.hstack([np.full(mask.sum(), 3)[:, np.newaxis],
                                vertex_of_cell[:, mask].transpose()])
    cell_types = np.empty(mask.sum(), dtype=np.uint8)
    cell_types[:] = vtk.VTK_TRIANGLE
    grid_icon_lake = pv.UnstructuredGrid(tri_indices.ravel(), cell_types,
                                         vertices)

# -----------------------------------------------------------------------------
# Visualise data
# -----------------------------------------------------------------------------

# Colormap
clim = (z.min(), z.max())
# clim = (0.0, 3250.0)  # overwrite manually
cmap = terrain3d.auxiliary.truncate_colormap(cm.bukavu, (0.5, 1.0))
cmap = terrain3d.auxiliary.discretise_colormap(cmap, num_cols=100)

# Plot
pl = pv.Plotter(window_size=(1800, 800), shape=(1, 2), border=False)
camera_position = \
[(30756.010579915594, 30758.52688644648, 34491.010579915775),
 (-1.837179297581315e-10, 2.5163065307042416, 3735.0),
 (0.0, 0.0, 1.0)]
# -----------------------------------------------------------------------------
pl.subplot(0, 0)
col_bar_args = dict(height=0.25, vertical=True, position_x=0.8, position_y=0.1)
pl.add_mesh(grid_dem, scalars="Surface elevation", show_edges=False, cmap=cmap,
            clim=clim, scalar_bar_args=col_bar_args)
if show_lakes:
    pl.add_mesh(grid_dem_lake, color=cm.bukavu(0.3), show_edges=False)
# pl.set_background("black")  # type: ignore
pl.remove_scalar_bar()  # type: ignore
pt_obs = pv.Sphere(radius=radar_radius,
                   center=np.array([0.0, 0.0, radar_loc[2] * ter_exa_fac]))
pl.add_mesh(pt_obs, color=radar_colour, show_edges=False)
pl.camera_position = camera_position
# -----------------------------------------------------------------------------
pl.subplot(0, 1)
pl.add_mesh(grid_icon, scalars="Surface elevation", show_edges=True, cmap=cmap,
            clim=clim, scalar_bar_args=col_bar_args)
if show_lakes:
    pl.add_mesh(grid_icon_lake, color=cm.bukavu(0.3), show_edges=True)
# pl.set_background("black")  # type: ignore
pl.remove_scalar_bar()  # type: ignore
pt_obs = pv.Sphere(radius=radar_radius,
                   center=np.array([0.0, 0.0, radar_loc[2] * ter_exa_fac]))
pl.add_mesh(pt_obs, color=radar_colour, show_edges=False)
pl.camera_position = camera_position
pl.link_views()
# -----------------------------------------------------------------------------
pl.show()
# pl.camera_position  # return camera position when plot is closed
