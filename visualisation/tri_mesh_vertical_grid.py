# Description: Visualise GEBCO or MERIT data for subregion in Switzerland with
#              a triangle mesh. Use a planar map projection and display
#              'idealised' vertical grid of a GCM/RCM. Optionally display lakes.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
from cmcrameri import cm
from matplotlib.colors import ListedColormap
import time
from pyproj import CRS
from pyproj import Transformer
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import terrain3d

mpl.style.use("classic")

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
    lon_ver, lat_ver, elevation_ver, crs_dem \
        = terrain3d.gebco.get(agg_num, domain)  # GEBCO
elif dem == "MERIT":
    lon_ver, lat_ver, elevation_ver, crs_dem = terrain3d.merit.get(domain)
    if agg_num > 1:
        lon_ver, lat_ver, elevation_ver \
            = terrain3d.auxiliary.aggregate_dem(lon_ver, lat_ver,
                                                elevation_ver, agg_num)
else:
    raise ValueError("Unknown DEM")

# Compute binary lake mask (optional)
if show_lakes:
    lon_quad = lon_ver[:-1] + np.diff(lon_ver) / 2.0
    lat_quad = lat_ver[:-1] + np.diff(lat_ver) / 2.0
    mask_lake = terrain3d.outlines.binary_mask(
        "shorelines", lon_quad, lat_quad, crs_dem, resolution="full", level=2,
        sub_sample_num=5)

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
quad_indices = terrain3d.triangles.get_quad_indices(len(lon_ver), len(lat_ver))

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

# Create wire frame that represent 'idealised' vertical grid of GCM/RCM
x_wire = np.arange(-55000.0, 80000.0, 25000.0, dtype=np.float32)
y_wire = np.arange(-30000.0, 65000.0, 25000.0, dtype=np.float32)
z_wire = np.arange(0.0, 20001.0, 5000.0, dtype=np.float32) * ter_exa_fac
x, y, z = np.meshgrid(x_wire, y_wire, z_wire)
wire_ent = pv.StructuredGrid(x, y, z).extract_all_edges()  # entire grid
slic = (slice(None), slice(None), slice(0, 2))  # lowest boxes
wire_low = pv.StructuredGrid(x[slic], y[slic], z[slic]).extract_all_edges()
slic = (slice(-2, None), slice(-2, None), slice(None))  # upper right
wire_ur = pv.StructuredGrid(x[slic], y[slic], z[slic]).extract_all_edges()

# Colormap
cmap = terrain3d.auxiliary.terrain_colormap(elevation_ver)

# Plot
pl = pv.Plotter(window_size=[1000, 1000])
col_bar_args = dict(height=0.25, vertical=True, position_x=0.8, position_y=0.1)
pl.add_mesh(grid, scalars="Surface elevation", show_edges=False, cmap=cmap,
            scalar_bar_args=col_bar_args)
if show_lakes:
    pl.add_mesh(grid_lake, color=cm.bukavu(0.3), show_edges=False)
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
