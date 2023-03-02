# Description: Visualise COSMO topography for the Hengduan Mountains
#              (Southeastern Tibetan Plateau) with 'grid cell columns'. Plot
#              three different topographies (present-day, reduced and
#              envelope).
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
import xarray as xr
from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Path for example data
path_examp = os.path.join(os.path.split(
    os.path.dirname(terrain3d.__file__))[0], "example_data/")

# Settings
terrain_exag_fac = 15.0  # terrain exaggeration factor [-]
topos = ["topo", "red_topo", "env_topo"]

# Load topographies
hsurf = {}
fr_land = {}
for i in topos:
    ds = xr.open_dataset(path_examp + "hengduan_mountains_4.4km_" + i + ".nc")
    # ds = ds.isel(rlon=slice(150, 670), rlat=slice(100, 610))  # default
    ds = ds.isel(rlon=slice(170, 640), rlat=slice(120, 570))  # smaller
    # ds = ds.isel(rlon=slice(240, 560), rlat=slice(180, 530))  # smallest
    hsurf[i] = ds["HSURF"].values.squeeze()  # (lat, rlon) [m]
    fr_land[i] = ds["FR_LAND"].values.squeeze() # (lat, rlon) [-]
    if i == topos[0]:
        print("Domain size: " + str(ds["HSURF"].shape))
        rlon = ds["rlon"].values
        rlat = ds["rlat"].values
    ds.close()

# Plot maximal range of elevation
hsurf_all = np.concatenate([hsurf[i][np.newaxis, :, :] for i in topos],
                           axis=0)
hsurf_rang = hsurf_all.max(axis=0) - hsurf_all.min(axis=0)
levels = np.arange(0.0, 4000.0, 500.0)
cmap = plt.get_cmap("Spectral")
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
plt.figure()
plt.pcolormesh(rlon, rlat, hsurf_rang, cmap=cmap, norm=norm)
plt.axis([rlon[0], rlon[-1], rlat[0], rlat[-1]])
plt.title("Maximal elevation range [m]", y=1.01, fontsize=12)
plt.colorbar()

# Compute edge coordinates
rlon_edge, rlat_edge = terrain3d.auxiliary.gridcoord(rlon, rlat)

# Compute vertices coordinates and terrain exaggeration
rad_earth = 6370997.0  # earth radius [m]
deg2m = (2.0 * np.pi * rad_earth) / 360.0  # [m deg-1]
x_ver = (rlon_edge * deg2m).astype(np.float64)
y_ver = (rlat_edge * deg2m).astype(np.float64)

# Compute visualisation data for different topographies
data = {}
for i in topos:

    # Pad elevation array with 0.0 at all sides
    elevation = hsurf[i] * terrain_exag_fac
    elevation_pad_0 = np.pad(elevation, [(1, 1), (1, 1)], mode="constant",
                             constant_values=np.minimum(0.0, elevation.min()))
    elevation_pad_0 = elevation_pad_0.clip(min=0.0)
    # visualised elevation -> clip values below sea level to 0.0 m
    # (-> 'cell_data', which is used for colouring, uses unmodified
    # 'elevation')

    # Compute vertices for grid cell columns
    vertices = terrain3d.columns.get_vertices(x_ver, y_ver, elevation_pad_0)
    shp_ver = vertices.shape
    vertices_rshp = vertices.reshape((y_ver.size * x_ver.size * 4), 3)

    # Compute quads for grid cell columns
    quads, cell_data, column_index \
        = terrain3d.columns.get_quads(elevation, elevation_pad_0, shp_ver)

    # Mask water grid cells (-> represent as blue area)
    mask_water = fr_land[i] < 0.5
    ind_ma_0, ind_ma_1 = np.where(mask_water)
    ind_ma = np.ravel_multi_index((ind_ma_0, ind_ma_1), elevation.shape)
    mask_1d = np.zeros(quads.shape[0], dtype=bool)
    for j in ind_ma:
        mask_1d[:column_index.size][column_index == j] = True

    # Main columns
    quads_sel = quads[~mask_1d, :]
    cell_data_sel = cell_data[~mask_1d]
    cell_types = np.empty(quads_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid = pv.UnstructuredGrid(quads_sel.ravel(), cell_types, vertices_rshp)
    grid.cell_data["Surface elevation [m]"] = cell_data_sel

    # Water columns
    quads_sel = quads[mask_1d, :]
    cell_types = np.empty(quads_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_water = pv.UnstructuredGrid(quads_sel.ravel(), cell_types,
                                     vertices_rshp)

    data[i] = {"grid": grid, "grid_water": grid_water,
               "mask_water": mask_water}

# -----------------------------------------------------------------------------
# Visualise data
# -----------------------------------------------------------------------------

# Colors
color_water = cm.bukavu(0.3)
colormap = terrain3d.auxiliary.truncate_colormap(cm.bukavu, (0.5, 1.0))
# colormap = terrain3d.auxiliary.ncl_colormap("OceanLakeLandSnow")
# color_water = colormap(0.0)
# colormap = terrain3d.auxiliary.truncate_colormap(colormap, (0.05, 1.0))

# Plot
names = {"topo": "Present-day", "red_topo": "Reduced", "env_topo": "Envelope"}
pl = pv.Plotter(window_size=(4750, 1100), shape=(1, 3))
for ind, i in enumerate(topos):
    pl.subplot(0, ind)
    pl.add_mesh(data[i]["grid"], cmap=colormap, show_edges=False, label="1",
                edge_color="black", line_width=5, show_scalar_bar=False)
    if np.any(data[i]["mask_water"]):
        pl.add_mesh(data[i]["grid_water"], color=color_water, show_edges=False,
                    label="1", edge_color="black", line_width=5)
    pl.add_text(names[i] + " topography", font_size=30, color="white",
                position=(150.0, 1000.0))
    pl.set_background("black")
    # pl.add_text(names[i], font_size=20, color="black")
    # pl.set_background("white")
pl.link_views()
pl.camera_position = \
[(-1518771.653528124, -2352702.0450042486, 3064710.2674981747),
 (-1474443.984375, -10007.5, 46759.09375),
 (-0.0028615219048856366, 0.7899525791238017, 0.6131612629871865)]
pl.show()
# pl.camera_position  # return camera position when plot is closed
