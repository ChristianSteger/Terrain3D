# Description: Visualise GEBCO data set for Switzerland with rectangular
#              columns (-> terrain representation in climate models). The
#              elevation of grid cells, which are below sea level and are land
#              according to the GSHHG data base, are set to 0.0 m. Lakes are
#              additionally displayed. Different spatial resolutions are
#              visualised.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
import xesmf as xe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from pyproj import CRS, Transformer
from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import time
from skimage.measure import label
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

pole_lat = 43.0
pole_lon = -170.0
cent_rot_lon = 0.0
d_spac = {"50km": 0.44, "12km": 0.11, "2km": 0.02}  # grid spacing [degree]
rlon_rang = (-2.95, 0.57)  # ~ range in rotated longitude direction [degree]
rlat_rang = (-1.28, 0.92)  # ~ range in rotated latitude direction [degree]
terrain_exag_fac = 6.0
depth_limit = -1500.0
gebco_agg_num = {"50km": 50, "12km": 12, "2km": 2}
frame = None  # None, "monochrome", "ocean"
lake_elev_equal = True  # equalise elevation of neighbouring lake grid cells
color_background = "black"  # "black", "white"
path_out = os.getenv("HOME") + "/Desktop/PyVista_gebco_res_ch/"
# output directory

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise ValueError("Output directory does not exist")

# Check background color and set linked variables
if color_background not in ("black", "white"):
    raise ValueError("Invalid background color for plot")
if color_background == "black":
    color_inverse = "white"
else:
    color_inverse = "black"

# Compute spatial extent of domain with coarsest resolution
crs_rot = CRS.from_user_input(
    ccrs.RotatedPole(pole_latitude=pole_lat,
                     pole_longitude=pole_lon,
                     central_rotated_longitude=cent_rot_lon)
)

# Compute rotated coordinates for different resolutions
rlon_cen, rlat_cen = np.mean(rlon_rang), np.mean(rlat_rang)
rot_coords = {}
for i in list(d_spac.keys()):
    rlon_len = round((rlon_rang[1] - rlon_rang[0]) / d_spac[i])
    rlat_len = round((rlat_rang[1] - rlat_rang[0]) / d_spac[i])
    rlon_edge = np.linspace(0.0, rlon_len * d_spac[i], rlon_len + 1)
    rlon_edge -=  (rlon_edge.mean() - rlon_cen)
    rlat_edge = np.linspace(0.0, rlat_len * d_spac[i], rlat_len + 1)
    rlat_edge -= (rlat_edge.mean() - rlat_cen)
    rlon = rlon_edge[:-1] + np.diff(rlon_edge) / 2.0
    rlat = rlat_edge[:-1] + np.diff(rlat_edge) / 2.0
    rot_coords[i] = {"rlon": rlon, "rlat": rlat,
                     "rlon_edge": rlon_edge, "rlat_edge": rlat_edge}

# # Plot extent of select domain
# line_prop = ({"color": "grey", "lw": 6.0, "ls": "-"},
#              {"color": "blue", "lw": 3.0, "ls": "--"},
#              {"color": "red", "lw": 1.0, "ls": "-"})
# plt.figure()
# ax = plt.axes(projection=crs_rot)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS)
# for ind, i in enumerate(list(rot_coords.keys())):
#     rlon_edge = rot_coords[i]["rlon_edge"]
#     rlat_edge = rot_coords[i]["rlat_edge"]
#     poly = [(rlon_edge[0], rlat_edge[0]), (rlon_edge[-1], rlat_edge[0]),
#             (rlon_edge[-1], rlat_edge[-1]), (rlon_edge[0], rlat_edge[-1])]
#     polygon = plt.Polygon(poly, facecolor="none",
#                           edgecolor=line_prop[ind]["color"],
#                           linewidth=line_prop[ind]["lw"],
#                           linestyle=line_prop[ind]["ls"],)
#     ax.add_patch(polygon)
# ax.set_extent([rlon_edge[0] - 0.5, rlon_edge[-1] + 0.5,
#                rlat_edge[0] - 0.5, rlat_edge[-1] + 0.5])

# Compute visualisation data for different topographies
data = {}
for i in list(rot_coords.keys()):

    # Select resolution
    rlon = rot_coords[i]["rlon"]
    rlat = rot_coords[i]["rlat"]
    rlon_edge = rot_coords[i]["rlon_edge"]
    rlat_edge = rot_coords[i]["rlat_edge"]

    # Compute domain for required GEBCO data
    dom_gebco = terrain3d.auxiliary.domain_extend_geo_coord(
        rlon, rlat, crs_rot, bound_res=0.001, domain_ext=0.1)

    # Load GEBCO data
    lon_in, lat_in, elevation_in, crs_dem \
        = terrain3d.gebco.get(gebco_agg_num[i], domain=dom_gebco)

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
    mask_land = terrain3d.outlines.binary_mask("shorelines", rlon, rlat,
                                               crs_rot,
                                               resolution="intermediate",
                                               level=1,
                                               sub_sample_num=10,
                                               filter_polygons=True)
    mask = mask_land & (elevation_in < 0.0)
    print("Ocean-land-inconsistency: increase elevation of "
          + str(mask.sum()) + " grid cells to 0.0 m")
    elevation_in[mask] = 0.0
    # -> set elevation of land grid cells to a minimal value of 0.0 m

    # Display lakes (-> as blue areas in visualisation)
    mask_lake = terrain3d.outlines.binary_mask(
        "shorelines", rlon, rlat, crs_rot, resolution="full", level=2,
        sub_sample_num=10)
    if lake_elev_equal:
        lakes_con, num_labels = label(mask_lake.astype(int), background=0,
                                      connectivity=2, return_num=True)
        for j in range(num_labels):
            mask = (lakes_con == (j + 1))
            elevation_in[mask] = elevation_in[mask].mean()

    # Compute vertices coordinates and terrain exaggeration
    x_ver = rlon_edge * terrain3d.constants.deg2m
    y_ver = rlat_edge * terrain3d.constants.deg2m
    elevation = elevation_in * terrain_exag_fac
    depth_limit_scal = depth_limit * terrain_exag_fac

    # Pad elevation array with 0.0 at all sides
    elevation_pad_0 = np.pad(elevation, [(1, 1), (1, 1)], mode="constant",
                             constant_values=np.minimum(0.0, elevation.min()))
    elevation_pad_0 = elevation_pad_0.clip(min=0.0)

    # Compute vertices for grid cell columns
    vertices = terrain3d.rect_columns.get_vertices(x_ver, y_ver,
                                                   elevation_pad_0)
    shp_ver = vertices.shape
    vertices_rshp = vertices.reshape((y_ver.size * x_ver.size * 4), 3)

    # Compute quads for grid cell columns
    quads, cell_data, column_index \
        = terrain3d.rect_columns.get_quads(elevation, elevation_pad_0, shp_ver)
    # -> 'cell_data', which is used for coloring, uses un-clipped elevation

    # Compute vertices/quads for frame (optional)
    if frame == "monochrome":
        vertices_rshp, quads_low = terrain3d.rect_columns \
            .add_frame_monochrome(depth_limit_scal, elevation, x_ver, y_ver,
                                  vertices_rshp, shp_ver)
    elif frame == "ocean":
        vertices_rshp, quads_ocean, cell_data_ocean, quads_low \
            = terrain3d.rect_columns \
            .add_frame_ocean(depth_limit_scal, elevation, x_ver, y_ver,
                             vertices_rshp, shp_ver)
        quads = np.vstack((quads, quads_ocean))
        cell_data = np.append(cell_data, cell_data_ocean)

    # Mask lake grid cells (-> represent as blue area)
    if np.any(mask_lake) and (elevation[mask_lake].min() < 0.0):
        raise ValueError("Lakes can only cover land grid cells")
    ind_ma_0, ind_ma_1 = np.where(mask_lake)
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
    grid.cell_data["Surface elevation [m]"] \
        = (cell_data_sel / terrain_exag_fac)
    # -> save real (un-scaled) elevation in 'cell_data'

    # Lake columns
    quads_sel = quads[mask_1d, :]
    cell_types = np.empty(quads_sel.shape[0], dtype=np.uint8)
    cell_types[:] = vtk.VTK_QUAD
    grid_lake = pv.UnstructuredGrid(quads_sel.ravel(), cell_types,
                                    vertices_rshp)

    data[i] = {"grid": grid, "grid_lake": grid_lake, "mask_lake": mask_lake,
               "cell_data_range": (cell_data_sel.min(), cell_data_sel.max())}

    # Frame quads (optional)
    if frame in ("monochrome", "ocean"):
        cell_types = np.empty(quads_low.shape[0], dtype=np.uint8)
        cell_types[:] = vtk.VTK_QUAD
        grid_low = pv.UnstructuredGrid(quads_low.ravel(), cell_types,
                                       vertices_rshp)

        data[i]["grid_low"] = grid_low

# -----------------------------------------------------------------------------
# Visualise data
# -----------------------------------------------------------------------------

# Colors
cell_data_min = np.array([data[i]["cell_data_range"][0]
                          for i in rot_coords.keys()]).min()
cell_data_max = np.array([data[i]["cell_data_range"][1]
                          for i in rot_coords.keys()]).max()
clim = (cell_data_min / terrain_exag_fac, cell_data_max / terrain_exag_fac)
clim = (0.0, 4500.0)  # overwrite manually
cmap = terrain3d.auxiliary.truncate_colormap(cm.bukavu, (0.5, 1.0))
cmap = terrain3d.auxiliary.discretise_colormap(cmap, num_cols=18)
color_lake = cm.bukavu(0.3)

# Plot
pos = ((0, 1), (1, 1), (1, 3))
groups = [(0, slice(1, 3)), (1, slice(0, 2)), (1, slice(2, 4))]
pl = pv.Plotter(window_size=(4000, 2500), shape=(2, 4), groups=groups,
                border=False, off_screen=True)
for ind, i in enumerate(list(rot_coords.keys())):
    pl.subplot(*pos[ind])
    col_bar_args = dict(height=0.60, vertical=True, position_x=0.93,
                        position_y=0.25, fmt="%.0f", label_font_size=35,
                        n_labels=10, title="", color=color_inverse,
                        italic=False, font_family="arial")
    pl.add_mesh(data[i]["grid"], cmap=cmap, clim=clim, show_edges=False,
                scalar_bar_args=col_bar_args)
    # pl.remove_scalar_bar()
    if np.any(data[i]["mask_lake"]):
        pl.add_mesh(data[i]["grid_lake"], color=color_lake, show_edges=False)
    if frame in ("monochrome", "ocean"):
        pl.add_mesh(data[i]["grid_low"], color="lightgrey", show_edges=False)
    txt = "~" + i[:-2] + " " + i[-2:]
    pl.add_text(txt, font_size=25, color=color_inverse,
                position=(260.0, 1020.0))
    pl.set_background(color_background)
pl.link_views()
pl.camera_position \
    = [(-149896.08171727188, -339872.08789620723, 490354.83265173994),
       (-124538.25919884289, -20015.07737124261, 10167.4248046875),
       (-0.012912707909070195, 0.832509657204179, 0.5538600298236325)]
pl.screenshot(path_out + "resolutions_ch.png")  # set off_screen=True
# pl.show()  # set off_screen=False
# pl.camera_position  # return camera position when plot is closed
pl.close()

# Plot inset map (optional)
linewidth = 1.5
labels_size = 14
labels_weight = "normal"  # "normal", "bold"
fig = plt.figure()
fig.patch.set_facecolor(color_background)
ax = plt.axes(projection=crs_rot)
ax.set_facecolor(color_background)
ax.add_feature(cfeature.COASTLINE, linewidth=linewidth, color=color_inverse)
ax.add_feature(cfeature.BORDERS, linewidth=linewidth, color=color_inverse)
rlon_edge = rot_coords["50km"]["rlon_edge"]
rlat_edge = rot_coords["50km"]["rlat_edge"]
xticks = np.arange(30.0, 50.0, 0.5)
yticks = np.arange(0.0, 15.0, 1.0)
gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl.right_labels = False
gl.top_labels = False
gl.xlines = False
gl.ylines = False
gl.ylocator = mticker.FixedLocator(xticks)
gl.xlocator = mticker.FixedLocator(yticks)
gl.xlabel_style = {"size": labels_size, "color": color_inverse,
                   "weight": labels_weight}
gl.ylabel_style = {"size": labels_size, "color": color_inverse,
                   "weight": labels_weight}
ax.spines["geo"].set_edgecolor(color_inverse)
ax.spines["geo"].set_linewidth(1.0)
ax.tick_params(axis="y", colors=color_inverse)
ax.yaxis.set_ticks_position("left")
ax.set_extent([rlon_edge[0], rlon_edge[-1], rlat_edge[0], rlat_edge[-1]],
              crs=crs_rot)
fig.savefig(path_out + "inset_map.png", facecolor=fig.get_facecolor(),
            dpi=175,  bbox_inches="tight")
plt.close()
