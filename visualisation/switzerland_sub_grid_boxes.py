# Description: Visualise GEBCO or MERIT data for subregion in Switzerland with
#              a triangle mesh. Use a planar map projection and display
#              'idealised' grid boxes of a GCM/RCM. Optionally display lakes.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys
import numpy as np
import vtk
import pyvista as pv
from cmcrameri import cm
from matplotlib.colors import ListedColormap
import time
from pyproj import CRS
from pyproj import Transformer
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# mpl.style.use("classic")

# Load required functions
sys.path.append("/Users/csteger/Downloads/Terrain3D/functions/")
from gebco import get as get_gebco
from merit import get as get_merit
from outlines import binary_mask as binary_mask_outlines
from auxiliary import get_quad_indices
from auxiliary import aggregate_dem

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

dem = "MERIT"  # GEBCO or MERIT
agg_num = 1  # spatial aggregation number for input DEM data
ter_exa_fac = 2.0  # terrain exaggeration factor [-]
show_lakes = True  # represent lakes as blue areas

# domain = (6.4, 8.5, 45.7, 46.7)  # Valais
domain = (7.9, 10.0, 46.25, 47.25)  # Central/Eastern Switzerland

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Get data (GEBCO or MERIT)
if dem == "GEBCO":
    lon_ver, lat_ver, elevation_ver = get_gebco(agg_num, domain)  # GEBCO
elif dem == "MERIT":
    lon_ver, lat_ver, elevation_ver = get_merit(domain)
    if agg_num > 1:
        lon_ver, lat_ver, elevation_ver \
            = aggregate_dem(lon_ver, lat_ver, elevation_ver, agg_num)
else:
    raise ValueError("Unknown DEM")


# Compute binary lake mask (optional)
if show_lakes:
    lon_quad = lon_ver[:-1] + np.diff(lon_ver) / 2.0
    lat_quad = lat_ver[:-1] + np.diff(lat_ver) / 2.0
    mask_lake = binary_mask_outlines("shorelines", lon_quad, lat_quad,
                                     resolution="full", level=2)

    # # Test plot
    # plt.figure()
    # plt.pcolormesh(lon_quad, lat_quad, mask_lake, shading="auto")
    # plt.colorbar()

# Transform geographic coordinates to orthographic projection
crs_wgs84 = CRS.from_string("EPSG:4326")
crs_ortho = CRS.from_dict({"proj": "ortho", "lat_0": lat_ver.mean(),
                           "lon_0": lon_ver.mean(), "ellps": "WGS84"})
transformer = Transformer.from_crs(crs_wgs84, crs_ortho, always_xy=True)
x, y, z = transformer.transform(*np.meshgrid(lon_ver, lat_ver),
                                elevation_ver * ter_exa_fac)

# # Test plot
# plt.figure()
# plt.pcolormesh(x, y, elevation_ver, shading="auto")
# plt.colorbar()

# Create indices array for quad vertices
num_quad_x = len(lon_ver) - 1
num_quad_y = len(lat_ver) - 1
quad_indices = get_quad_indices(len(lon_ver), len(lat_ver))

# Reshape arrays
vertices = np.hstack((x.ravel()[:, np.newaxis],
                      y.ravel()[:, np.newaxis],
                      z.ravel()[:, np.newaxis]))
quad_indices = quad_indices.reshape(num_quad_y * num_quad_x, 5)
if quad_indices.max() >= vertices.shape[0]:
    raise ValueError("Index out of bounds!")

# Create mesh for terrain
if not show_lakes:
    cell_types = np.empty(quad_indices.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid = pv.UnstructuredGrid(quad_indices.ravel(), cell_types, vertices)
    grid.point_data["Surface elevation"] = elevation_ver.ravel()
else:
    mask_lake_rav = mask_lake.ravel()
    quad_sel = quad_indices[~mask_lake_rav, :]
    cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)
    grid.point_data["Surface elevation"] = elevation_ver.ravel()

# Create mesh for lake-covered area (optional)
if show_lakes:
    quad_sel = quad_indices[mask_lake_rav, :]
    cell_types = np.empty(quad_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_lake = pv.UnstructuredGrid(quad_sel.ravel(), cell_types, vertices)

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

# Create 'idealised' wire frame that represent GCM/RCM grid
x_wire = np.arange(-55000.0, 80000.0, 25000.0, dtype=np.float32)
y_wire = np.arange(-30000.0, 65000.0, 25000.0, dtype=np.float32)
z_wire = np.arange(0.0, 20001.0, 5000.0, dtype=np.float32) * ter_exa_fac
x, y, z = np.meshgrid(x_wire, y_wire, z_wire)
wire_ent = pv.StructuredGrid(x, y, z).extract_all_edges()  # entire grid
slic = (slice(None), slice(None), slice(0, 2))  # lowest boxes
wire_low = pv.StructuredGrid(x[slic], y[slic], z[slic]).extract_all_edges()
slic = (slice(-2, None), slice(-2, None), slice(None))  # upper right
wire_ur = pv.StructuredGrid(x[slic], y[slic], z[slic]).extract_all_edges()

# Plot
pl = pv.Plotter(window_size=[1000, 1000])
col_bar_args = dict(height=0.25, vertical=True, position_x=0.8, position_y=0.1)
pl.add_mesh(grid, scalars="Surface elevation", show_edges=False, label="1",
            edge_color="black", line_width=0, cmap=colormap,
            scalar_bar_args=col_bar_args)
if show_lakes:
    pl.add_mesh(grid_lake, color=cm.bukavu(0.3), show_edges=False, label="1",
                edge_color="black", line_width=0)
pl.add_mesh(wire_ent, show_edges=True, style="wireframe", line_width=5.0,
            color="grey", edge_color="white", opacity=0.2)
pl.add_mesh(wire_low, show_edges=True, style="wireframe", line_width=5.0,
            color="grey", edge_color="white", opacity=0.8)
pl.add_mesh(wire_ur, show_edges=True, style="wireframe", line_width=5.0,
            color="grey", edge_color="white", opacity=0.8)
pl.set_background("black")
pl.remove_scalar_bar()
pl.camera_position = \
    [(-62054.57748529176, -155350.56959819535, 233420.00951741208),
     (1.3096723705530167e-10, 266.91871005502617, 24000.0),
     (0.06064708482631294, 0.7924907477737412, 0.60686105971226)]
pl.show()
# pl.camera_position  # return camera position when plot is closed
