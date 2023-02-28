# Description: Visualise GEBCO data set with 'grid cell columns' (-> terrain
#              representation in climate models). The elevation of grid cells,
#              which are below sea level and are land according to the GSHHG
#              data base, are set to 0.0 m. Lakes can optionally be displayed.
#              Different spatial resolutions are visualised.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
path_esmf = "/Users/csteger/miniconda3/envs/pyvista/lib/esmf.mk" # MacBook
# path_esmf = "/Users/csteger/opt/miniconda3/envs/pyvista/lib/esmf.mk" # Deskt.
os.environ["ESMFMKFILE"] = path_esmf
import xesmf as xe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from pyproj import CRS, Transformer
from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from skimage.measure import label
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Switzerland (~50 km, ~12 km, ~2 km)
pole_lat = 43.0
pole_lon = -170.0
cent_rot_lon = 0.0
d_rlon = {"50km": 0.44, "12km": 0.11, "2km": 0.02}
d_rlat = {"50km": 0.44, "12km": 0.11, "2km": 0.02}
rlon_0 = -3.1  # only for coarsest res. -> others automatically determined
rlat_0 = -1.5  # only for coarsest res. -> others automatically determined
rlon_len = 10 # only for coarsest res. -> others automatically determined
rlat_len = 7 # only for coarsest res. -> others automatically determined
terrain_exag_fac = 5.0
depth_limit = -1500.0
gebco_agg_num = {"50km": 50, "12km": 12, "2km": 2}
frame = None
show_lakes = True

# # Middle/South Europe (~200 km, ~50 km, ~12 km)
# pole_lat = 43.0
# pole_lon = -170.0
# cent_rot_lon = 0.0
# d_rlon = 0.33
# d_rlat = 0.33
# rlon_0 = -16.0
# rlat_0 = -11.3
# rlon_len = 100
# rlat_len = 90
# terrain_exag_fac = 40.0
# depth_limit = -5200.0
# gebco_agg_num = 36
# frame = "ocean"  # None, "monochrome", "ocean"
# show_lakes = False

# General settings
plot_sel_dom = True  # plot domain selection
lake_elev_equal = True  # equalise elevation of neighbouring lake grid cells

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Compute spatial extent of domain with coarsest resolution
crs_rot = CRS.from_user_input(
    ccrs.RotatedPole(pole_latitude=pole_lat,
                     pole_longitude=pole_lon,
                     central_rotated_longitude=cent_rot_lon)
)
res_co = list(d_rlon.keys())[0]  # coarset resolution
rlon_lim = (rlon_0 - d_rlon[res_co] / 2.0,
            rlon_0 + (rlon_len - 1) * d_rlon[res_co] + d_rlon[res_co] / 2.0)
rlat_lim = (rlat_0 - d_rlat[res_co] / 2.0,
            rlat_0 + (rlat_len - 1) * d_rlat[res_co] + d_rlat[res_co] / 2.0)

# Check extent of select domain (coarsest resolution)
if plot_sel_dom:
    plt.figure()
    ax = plt.axes(projection=crs_rot)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    poly = [(rlon_lim[0], rlat_lim[0]), (rlon_lim[1], rlat_lim[0]),
            (rlon_lim[1], rlat_lim[1]), (rlon_lim[0], rlat_lim[1])]
    polygon = plt.Polygon(poly, facecolor="lightblue", edgecolor="blue",
                          linewidth=2.5)
    ax.add_patch(polygon)
    ax.set_extent([rlon_lim[0] - 0.5, rlon_lim[1] + 0.5,
                   rlat_lim[0] - 0.5, rlat_lim[1] + 0.5])

# Compute rotated coordinates for different resolutions
rlon_res, rlat_res = {}, {}
rlon_res["50km"] = np.linspace(rlon_0,
                               rlon_0 + (rlon_len - 1) * d_rlon["50km"],
                               rlon_len)
rlat_res["50km"] = np.linspace(rlat_0,
                               rlat_0 + (rlat_len - 1) * d_rlat["50km"],
                               rlat_len)
for i in list(d_rlon.keys())[1:]:
    num_rlon = int(np.round((d_rlon[res_co] / d_rlon[i]) * rlon_len,
                            decimals=0))
    num_rlat = int(np.round((d_rlat[res_co] / d_rlat[i]) * rlat_len,
                            decimals=0))
    rlon_res[i] = np.linspace(rlon_0, rlon_0 + (num_rlon - 1) * d_rlon[i],
                          num_rlon)
    rlat_res[i] = np.linspace(rlat_0, rlat_0 + (num_rlat - 1) * d_rlat[i],
                           num_rlat)
    rlon_res[i] += (rlon_res["50km"].mean() - rlon_res[i].mean())
    rlat_res[i] += (rlat_res["50km"].mean() - rlat_res[i].mean())

# Select resolution
res = "2km"
rlon = rlon_res[res]
rlat = rlat_res[res]
rlon_edge, rlat_edge = terrain3d.auxiliary.gridcoord(rlon, rlat)

# Compute domain for required GEBCO data
dom_gebco = terrain3d.auxiliary.domain_extend_geo_coord(
    rlon, rlat, crs_rot, bound_res=0.001, domain_ext=0.1)

# Load GEBCO data
lon_in, lat_in, elevation_in, crs_dem = terrain3d.gebco.get(gebco_agg_num[res],
                                                            domain=dom_gebco)

# Remap elevation data to rotated grid
lon_in_edge, lat_in_edge = terrain3d.auxiliary.gridcoord(lon_in, lat_in)
grid_in = xr.Dataset({"lat": (["lat"], lat_in),
                      "lon": (["lon"], lon_in),
                      "lat_b": (["lat_b"], lat_in_edge),
                      "lon_b": (["lon_b"], lon_in_edge)})
transformer = Transformer.from_crs(crs_rot, crs_dem,
                                   always_xy=True)
lon_out, lat_out = transformer.transform(*np.meshgrid(rlon, rlat))
lon_edge_out, lat_edge_out = transformer.transform(*np.meshgrid(rlon_edge,
                                                                rlat_edge))
grid_out = xr.Dataset({"lat": (["y", "x"], lat_out),
                       "lon": (["y", "x"], lon_out),
                       "lat_b": (["y_b", "x_b"], lat_edge_out),
                       "lon_b": (["y_b", "x_b"], lon_edge_out)})
print("Remap GEBCO data to rotated grid")
t_beg = time.time()
regridder = xe.Regridder(grid_in, grid_out, "conservative")
elevation_in = regridder(elevation_in.astype(np.float32))
print("Elapsed time: %.1f" % (time.time() - t_beg) + " s")

# Compute land-sea mask
mask_land = terrain3d.outlines.binary_mask("shorelines", rlon, rlat, crs_rot,
                                           resolution="intermediate", level=1,
                                           sub_sample_num=10,
                                           filter_polygons=True)
mask = mask_land & (elevation_in < 0.0)
print("Ocean-land-inconsistency: increase elevation of " + str(mask.sum())
      + " grid cells to 0.0 m")
elevation_in[mask] = 0.0
# -> set elevation of land grid cells to a minimal value of 0.0 m

# Display lakes (-> as blue areas in visualisation)
if not show_lakes:
    mask_lake = np.zeros(elevation_in.shape, dtype=bool)
else:
    mask_lake = terrain3d.outlines.binary_mask(
        "shorelines", rlon, rlat, crs_rot, resolution="full", level=2,
        sub_sample_num=10)
    if lake_elev_equal:
        lakes_con, num_labels = label(mask_lake.astype(int), background=0,
                                      connectivity=2, return_num=True)
        for i in range(num_labels):
            mask = (lakes_con == (i + 1))
            elevation_in[mask] = elevation_in[mask].mean()

# Compute vertices coordinates and terrain exaggeration
rad_earth = 6370997.0  # earth radius [m]
deg2m = (2.0 * np.pi * rad_earth) / 360.0  # [m deg-1]
x_ver = rlon_edge * deg2m
y_ver = rlat_edge * deg2m
elevation = elevation_in * terrain_exag_fac
depth_limit *= terrain_exag_fac

# Pad elevation array with 0.0 at all sides
elevation_pad_0 = np.pad(elevation, [(1, 1), (1, 1)], mode="constant",
                         constant_values=np.minimum(0.0, elevation.min()))
elevation_pad_0 = elevation_pad_0.clip(min=0.0)
# visualised elevation -> clip values below sea level to 0.0 m
# (-> 'cell_data', which is used for colouring, uses unmodified 'elevation')

# Compute vertices for grid cell columns
vertices = terrain3d.columns.get_vertices(x_ver, y_ver, elevation_pad_0)
shp_ver = vertices.shape
vertices_rshp = vertices.reshape((y_ver.size * x_ver.size * 4), 3)

# Compute quads for grid cell columns
quads, cell_data, column_index \
    = terrain3d.columns.get_quads(elevation, elevation_pad_0, shp_ver)

# Compute vertices/quads for frame (optional)
if frame == "monochrome":
    vertices_rshp, quads_low \
        = terrain3d.columns.add_frame_monochrome(depth_limit, elevation, x_ver,
                                                 y_ver, vertices_rshp, shp_ver)
elif frame == "ocean":
    vertices_rshp, quads_ocean, cell_data_ocean, quads_low \
        = terrain3d.columns.add_frame_ocean(depth_limit, elevation, x_ver,
                                            y_ver, vertices_rshp, shp_ver)
    quads = np.vstack((quads, quads_ocean))
    cell_data = np.append(cell_data, cell_data_ocean)

# Mask lake grid cells (-> represent as blue area)
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

colormap = terrain3d.auxiliary.cmap_terrain(elevation, cm.bukavu)
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
# pl.camera_position = \
#     [(-282445.58202424366, -540119.7098944773, 637043.5660801955),
#      (-136769.69537015786, -20571.05174266603, 7556.56201171875),
#      (0.10358509640148689, 0.75519009904462, 0.6472696826736686)]
pl.show()
# pl.camera_position  # return camera position when plot is closed
