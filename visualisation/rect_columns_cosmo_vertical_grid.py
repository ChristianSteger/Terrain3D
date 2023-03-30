# Description: Visualise COSMO topography for a subregion of the Alps with
#              rectangular columns. Vertical height-based hybrid (Gal-Chen)
#              coordinates are additionally represented.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
import cartopy.crs as ccrs
import xarray as xr
from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Select COSMO resolution and domain
# -----------------------------------------------------------------------------

# Download example data
path_examp = terrain3d.auxiliary.get_path_data() + "example_data/"
if not os.path.exists(path_examp):
    os.makedirs(path_examp)
for i in ["greater_alpine_region_2.2km.nc", "europe_12km.nc"]:
    terrain3d.auxiliary.download_file(
        "https://github.com/ChristianSteger/Terrain3D/blob/main/example_data/"
        + i + "?raw=true", path_examp + i)

# # ~2.2 km
# ds = xr.open_dataset(path_examp + "greater_alpine_region_2.2km.nc")
# ds = ds.isel(rlon=slice(370, 420), rlat=slice(415, 455))
# hsurf = ds["HSURF"].values.squeeze()  # (lat, rlon) [m]
# fr_land = ds["FR_LAND"].values.squeeze() # (lat, rlon) [-]
# vcoord = ds["vcoord"].values  # (level1)
# vcflat = ds["vcoord"].vcflat
# rlon = ds["rlon"].values
# rlat = ds["rlat"].values
# ds.close()
# terrain_exag_fac = 4.0  # terrain exaggeration factor [-]
# sel_thin = (slice(0, 15), slice(0, 15), [5, 15, 30, 45, 60])
# sel_thick = (slice(0, 15), slice(0, 15), [15, 60])
# # selection of grid that is plotted with thin/thick lines (y, x, z)

# ~12 km
ds = xr.open_dataset(path_examp + "europe_12km.nc")
ds = ds.isel(rlon=slice(185, 210), rlat=slice(155, 180))
hsurf = ds["HSURF"].values.squeeze()  # (lat, rlon) [m]
fr_land = ds["FR_LAND"].values.squeeze() # (lat, rlon) [-]
vcoord = ds["vcoord"].values  # (level1)
vcflat = ds["vcoord"].vcflat
rlon = ds["rlon"].values
rlat = ds["rlat"].values
ds.close()
terrain_exag_fac = 8.0  # terrain exaggeration factor [-]
sel_thin = (slice(7, 17), slice(3, 17), [5, 15, 30, 45, 60])
sel_thick = (slice(7, 17), slice(3, 17), [15, 60])

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# # Plot elevation of select domain
# plt.figure()
# plt.pcolormesh(rlon, rlat, hsurf)
# plt.colorbar()

# Compute edge coordinates
rlon_edge, rlat_edge = terrain3d.auxiliary.gridcoord(rlon, rlat)

# Compute vertices coordinates and terrain exaggeration
x_ver = (rlon_edge * terrain3d.constants.deg2m).astype(np.float64)
y_ver = (rlat_edge * terrain3d.constants.deg2m).astype(np.float64)
elevation = hsurf * terrain_exag_fac

# Pad elevation array with 0.0 at all sides
elevation_pad_0 = np.pad(elevation, [(1, 1), (1, 1)], mode="constant",
                         constant_values=np.minimum(0.0, elevation.min()))
elevation_pad_0 = elevation_pad_0.clip(min=0.0)

# Compute vertices for grid cell columns
vertices = terrain3d.rect_columns.get_vertices(x_ver, y_ver, elevation_pad_0)
shp_ver = vertices.shape
vertices_rshp = vertices.reshape((y_ver.size * x_ver.size * 4), 3)

# Compute quads for grid cell columns
quads, cell_data, column_index \
    = terrain3d.rect_columns.get_quads(elevation, elevation_pad_0, shp_ver)
# -> 'cell_data', which is used for coloring, uses un-clipped elevation

# Mask lake/ocean grid cells (-> represent as blue area)
mask_water = (fr_land < 0.5)
ind_ma_0, ind_ma_1 = np.where(mask_water)
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
grid.cell_data["Surface elevation [m]"] = (cell_data_sel / terrain_exag_fac)
# -> save real (un-scaled) elevation in 'cell_data'

# Lake/ocean columns
quads_sel = quads[mask_1d, :]
cell_types = np.empty(quads_sel.shape[0], dtype=np.uint8)
cell_types[:] = vtk.VTK_QUAD
grid_water = pv.UnstructuredGrid(quads_sel.ravel(), cell_types, vertices_rshp)

# Compute height-based hybrid (Gal-Chen) coordinates (grid cell centre)
z = np.empty(hsurf.shape + (len(vcoord),), dtype=np.float32)  # (y, x, z)
mask = (vcflat <= vcoord) & (vcoord <= vcoord[0])
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        z[i, j, mask] = vcoord[mask] + 0.0 * hsurf[i, j]
        z[i, j, ~mask] = vcoord[~mask] \
                         + (vcflat - vcoord[~mask]) / vcflat * hsurf[i, j]
z = (np.flip(z, axis=2) * terrain_exag_fac)

# Interpolate z-values at grid cell edges and compute x- and y-egde-coordinates
z_ip = np.empty((z.shape[0] - 1, z.shape[1] - 1, z.shape[2]), dtype=np.float32)
x = np.arange(z.shape[1], dtype=np.float32)
y = np.arange(z.shape[0], dtype=np.float32)
x_ip, y_ip = x[:-1] + np.diff(x) / 2.0, y[:-1] + np.diff(y) / 2.0
for i in range(z.shape[2]):
    f_ip = interpolate.RectBivariateSpline(y, x, z[:, :, i], kx=1, ky=1)
    z_ip[:, :, i] = f_ip(y_ip, x_ip)
x_2d, y_2d = np.meshgrid(x_ver[1:-1], y_ver[1:-1])
x = np.repeat(x_2d[:, :, np.newaxis], z.shape[2], axis=2)
y = np.repeat(y_2d[:, :, np.newaxis], z.shape[2], axis=2)

# Create wire frame that represent vertical grid of COSMO (and take subset)
wire_ent = pv.StructuredGrid(x[sel_thin], y[sel_thin],
                             z_ip[sel_thin]).extract_all_edges()
wire_thick = []
for i in sel_thick[2]:
    sel = (sel_thick[0], sel_thick[1], i)
    wire_thick.append(pv.StructuredGrid(x[sel], y[sel], z_ip[sel])
                      .extract_all_edges())

# -----------------------------------------------------------------------------
# Visualise data
# -----------------------------------------------------------------------------

# Colormap
cmap = terrain3d.auxiliary.truncate_colormap(cm.bukavu, (0.5, 1.0))
cmap = terrain3d.auxiliary.discretise_colormap(cmap, num_cols=16)
clim = (0.0, 4000.0)

# Plot
pl = pv.Plotter()
pl.add_mesh(grid, cmap=cmap, clim=clim, show_edges=False,
            show_scalar_bar=False)
if np.any(mask_water):
    pl.add_mesh(grid_water, color=cm.bukavu(0.3), show_edges=False)
pl.add_mesh(wire_ent, show_edges=True, style="wireframe", line_width=5.0,
            color="grey", edge_color="white", opacity=0.2)
for i in wire_thick:
    pl.add_mesh(i, show_edges=True, style="wireframe", line_width=5.0,
                color="grey", edge_color="white", opacity=0.8)
pl.set_background("black")
pl.show()
