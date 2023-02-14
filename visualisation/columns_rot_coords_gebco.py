# Description: Simple example to illustrate visualisation of terrain with 'grid
#              cell columns'
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import sys
import numpy as np
import vtk
import pyvista as pv
from cmcrameri import cm
from matplotlib.colors import ListedColormap

# Load required functions
sys.path.append("/Users/csteger/Downloads/Terrain3D/functions/")
from gebco import get as get_gebco
from columns import get_vertices, get_quads
from columns import add_frame_monochrome, add_frame_ocean

# -----------------------------------------------------------------------------
# General settings
# -----------------------------------------------------------------------------

# Plot frame around domain
frame = "ocean"  # None, "monochrome", "ocean"

# -----------------------------------------------------------------------------
# Artifical data
# -----------------------------------------------------------------------------

# # elevation = np.array([[-5, -2, 3, 4, -3, -5],
# #                       [-3, -1, 6, 8, 7, -6],
# #                       [-3, 1, 6, 2, 3, 9],
# #                       [-7, -2, 8, 2, -3, -4]], dtype=np.float32) * 0.2
# x_ver = np.arange(elevation.shape[1] + 1) * 1.33
# y_ver = np.arange(elevation.shape[0] + 1) * 1.33
# depth_limit = -1.6

# -----------------------------------------------------------------------------
# Real data
# -----------------------------------------------------------------------------

# Select domain
# domain = (5, 17, 43, 49)  # Alps
# domain = (-5, 40, 30, 60)  # Europe
domain = (90, 130, -15, 10)  # South-East Asia
# domain = (-80, -35, -45, -15)  # South America

# Miscellaneous settings
terrain_exag_fac = 15.0  # terrain exaggeration factor [-]
depth_limit = -150000.0

# Load and process data
lon, lat, elevation, crs_dem = get_gebco(10, domain)
elevation = elevation.astype(np.float32) * terrain_exag_fac
rad_earth = 6370997.0  # earth radius [m]
dx = (2.0 * np.pi * rad_earth * np.cos(np.deg2rad(lat.mean()))) / 360.0 \
     * np.diff(lon).mean()
dy = (2.0 * np.pi * rad_earth) / 360.0 * np.diff(lon).mean()
x = np.linspace(0.0, dx * (lon.size - 1), lon.size)
y = np.linspace(0.0, dy * (lat.size - 1), lat.size)
x_ver = np.hstack((x[0] - (dx / 2.), x[:-1] + np.diff(x) / 2.,
                   x[-1] + (dx / 2.)))
y_ver = np.hstack((y[0] - (dy / 2.), y[:-1] + np.diff(y) / 2.,
                   y[-1] + (dy / 2.)))

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Pad elevation array with 0.0 at all sides
elevation_pad_0 = np.pad(elevation, [(1, 1), (1, 1)], mode="constant",
                         constant_values=np.minimum(0.0, elevation.min()))
elevation_pad_0 = elevation_pad_0.clip(min=0.0)
# visualised elevation -> clip values below sea level to 0.0 m
# (-> 'cell_data', which is used for colouring, uses unmodified 'elevation')

# Compute vertices for grid cell columns
vertices = get_vertices(x_ver, y_ver, elevation_pad_0)
shp_ver = vertices.shape
vertices_rshp = vertices.reshape((y_ver.size * x_ver.size * 4), 3)

# Compute quads for grid cell columns
quads, cell_data, column_index = get_quads(elevation, elevation_pad_0, shp_ver)

# Compute vertices/quads for frame (optional)
if frame == "monochrome":
    vertices_rshp, quads_low \
        = add_frame_monochrome(depth_limit, elevation, x_ver, y_ver,
                               vertices_rshp, shp_ver)
elif frame == "ocean":
    vertices_rshp, quads_ocean, cell_data_ocean, quads_low \
        = add_frame_ocean(depth_limit, elevation, x_ver, y_ver, vertices_rshp,
                          shp_ver)
    quads = np.vstack((quads, quads_ocean))
    cell_data = np.append(cell_data, cell_data_ocean)

# Mask lake grid cells (-> represent as blue area)
mask_lake = np.zeros(elevation.shape, dtype=bool)
# mask_lake[80:90, 0:10] = True
# mask_lake[80:90, 80:100] = True
if np.any(mask_lake) and (elevation[mask_lake].min() < 0.0):
    raise ValueError("Lakes can only cover land grid cells")
ind_ma_0, ind_ma_1 = np.where(mask_lake)
ind_ma = np.ravel_multi_index((ind_ma_0, ind_ma_1), elevation.shape)
mask_1d = np.zeros(quads.shape[0], dtype=bool)
for i in ind_ma:
    mask_1d[:column_index.size][column_index == i] = True

# Main columns
quads_sel = quads[~mask_1d, :]
cell_data_sel = cell_data[~mask_1d]
cell_types = np.empty(quads_sel.shape[0], dtype=np.uint8)
cell_types[:] = vtk.VTK_QUAD
grid = pv.UnstructuredGrid(quads_sel.ravel(), cell_types, vertices_rshp)
grid.cell_data["Surface elevation [m]"] = cell_data_sel

# Lake columns
quads_sel = quads[mask_1d, :]
cell_types = np.empty(quads_sel.shape[0], dtype=np.uint8)
cell_types[:] = vtk.VTK_QUAD
grid_lake = pv.UnstructuredGrid(quads_sel.ravel(), cell_types, vertices_rshp)

# Frame quads (optional)
if frame in ("monochrome", "ocean"):
    cell_types = np.empty(quads_low.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_low = pv.UnstructuredGrid(quads_low.ravel(), cell_types,
                                   vertices_rshp)

# -----------------------------------------------------------------------------
# Visualise data
# -----------------------------------------------------------------------------

# Colormap
num_cols = 256
mapping = np.linspace(elevation.min(), elevation.max(), num_cols)
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
pl = pv.Plotter()
pl.add_mesh(grid, cmap=colormap, show_edges=False, label="1",
            edge_color="black", line_width=5, show_scalar_bar=False)
if np.any(mask_lake):
    pl.add_mesh(grid_lake, color=cm.bukavu(0.3), show_edges=False, label="1",
                edge_color="black", line_width=5)
pl.set_background("black")
if frame in ("monochrome", "ocean"):
    pl.add_mesh(grid_low, color="lightgrey", show_edges=False, label="1",
                edge_color="black", line_width=5)
pl.set_background("black")
pl.show()
