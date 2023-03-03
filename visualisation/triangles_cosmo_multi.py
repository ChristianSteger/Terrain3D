# Description: Visualise COSMO topography for the Hengduan Mountains
#              (Southeastern Tibetan Plateau) with a triangle mesh. Plot
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
from scipy import interpolate
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Path for example data
path_examp = os.path.join(os.path.split(
    os.path.dirname(terrain3d.__file__))[0], "example_data/")

# Settings
terrain_exag_fac = 12.0  # terrain exaggeration factor [-]
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

# # Plot maximal range of elevation
# hsurf_all = np.concatenate([hsurf[i][np.newaxis, :, :] for i in topos],
#                            axis=0)
# hsurf_rang = hsurf_all.max(axis=0) - hsurf_all.min(axis=0)
# levels = np.arange(0.0, 4000.0, 500.0)
# cmap = plt.get_cmap("Spectral")
# norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
# plt.figure()
# plt.pcolormesh(rlon, rlat, hsurf_rang, cmap=cmap, norm=norm)
# plt.axis([rlon[0], rlon[-1], rlat[0], rlat[-1]])
# plt.title("Maximal elevation range [m]", y=1.01, fontsize=12)
# plt.colorbar()

# Compute visualisation data for different topographies
data = {}
for i in topos:

    # Compute coordinates and elevation at edges (--> vertices)
    rlon_edge, rlat_edge = terrain3d.auxiliary.gridcoord(rlon, rlat)
    rlon_edge, rlat_edge = rlon_edge[1:-1], rlat_edge[1:-1]
    x = np.arange(hsurf[i].shape[1], dtype=np.float32)
    y = np.arange(hsurf[i].shape[0], dtype=np.float32)
    f_ip = interpolate.RectBivariateSpline(y, x, hsurf[i], kx=1, ky=1)
    x_ip = x[:-1] + np.diff(x) / 2.0
    y_ip = y[:-1] + np.diff(y) / 2.0
    hsurf_ip = f_ip(y_ip, x_ip)

    # Compute cartesian vertices coordinates and terrain exaggeration
    rad_earth = 6370997.0  # earth radius [m]
    deg2m = (2.0 * np.pi * rad_earth) / 360.0  # [m deg-1]
    x_ver = (rlon_edge * deg2m).astype(np.float64)
    y_ver = (rlat_edge * deg2m).astype(np.float64)
    x, y = np.meshgrid(x_ver, y_ver)
    z = hsurf_ip * terrain_exag_fac

    # Create indices array for quad vertices
    num_quad_x = len(x_ver) - 1
    num_quad_y = len(y_ver) - 1
    quad_indices = terrain3d.triangles.get_quad_indices(len(x_ver), len(y_ver))

    # Reshape arrays
    vertices = np.hstack((x.ravel()[:, np.newaxis],
                          y.ravel()[:, np.newaxis],
                          z.ravel()[:, np.newaxis]))
    quad_indices = quad_indices.reshape(num_quad_y * num_quad_x, 5)
    if quad_indices.max() >= vertices.shape[0]:
        raise ValueError("Index out of bounds!")

    # Create mesh for terrain
    mask_water = fr_land[i][1:-1, 1:-1] < 0.5
    mask_water_rav = mask_water.ravel()
    quad_sel = quad_indices[~mask_water_rav, :]
    cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)
    grid.point_data["Surface elevation"] = hsurf_ip.ravel()

    # Create mesh for water area
    quad_sel = quad_indices[mask_water_rav, :]
    cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_water = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)

    data[i] = {"grid": grid, "grid_water": grid_water}

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
pos = ((0, 1), (1, 1), (1, 3))
groups = [(0, slice(1, 3)), (1, slice(0, 2)), (1, slice(2, 4))]
pl = pv.Plotter(window_size=(2600, 1820), shape=(2, 4), groups=groups)
for ind, i in enumerate(topos):
    pl.subplot(*pos[ind])
    col_bar_args = dict(height=0.25, vertical=False, position_x=0.8,
                        position_y=0.1)
    pl.add_mesh(data[i]["grid"], scalars="Surface elevation", show_edges=False,
                label="1", edge_color="black", line_width=0, cmap=colormap,
                scalar_bar_args=col_bar_args)
    pl.add_mesh(data[i]["grid_water"], color=cm.bukavu(0.3), show_edges=False,
                label="1", edge_color="black", line_width=0)
    pl.remove_scalar_bar()
    pl.add_text(names[i] + " topography", font_size=20, color="white",
                position=(150.0, 800.0))
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
