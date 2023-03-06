# Description: Visualise GEBCO data set with rectangular columns (-> terrain
#              representation in climate models). The elevation of grid cells,
#              which are below sea level and are land according to the GSHHG
#              data base, are set to 0.0 m. Lakes can optionally be displayed.
#              Different spatial resolutions are visualised.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
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
import time
from skimage.measure import label
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# # Switzerland (~50 km, ~12 km, ~2 km)
# pole_lat = 43.0
# pole_lon = -170.0
# cent_rot_lon = 0.0
# d_spac = {"50km": 0.44, "12km": 0.11, "2km": 0.02}  # grid spacing [degree]
# rlon_rang = (-3.1, 0.86)  # ~ range in rotated longitude direction [degree]
# rlat_rang = (-1.5, 1.14) # ~ range in rotated latitude direction [degree]
# terrain_exag_fac = 6.0
# depth_limit = -1500.0
# gebco_agg_num = {"50km": 50, "12km": 12, "2km": 2}
# frame = None  # None, "monochrome", "ocean"
# show_lakes = True
# window_size = (3000, 2000)
# camera_position = \
#     [(-104360.44462933527, -275666.88535599434, 660212.3759366738),
#      (-124538.25919884289, -20015.07737124261, 10167.4248046875),
#      (0.02877473710164539, 0.9305350169623949, 0.3650706735846183)]
# position_txt = (150.0, 850.0) # position of sub-plot labels

# Middle/South Europe (~200 km, ~50 km, ~12 km)
pole_lat = 43.0
pole_lon = -170.0
cent_rot_lon = 0.0
d_spac = {"200km": 1.76, "50km": 0.44, "12km": 0.11}
rlon_rang = (-15.0, 15.0)
rlat_rang = (-11.0, 13.0)
terrain_exag_fac = 40.0
depth_limit = -5200.0
gebco_agg_num = {"200km": 100, "50km": 50, "12km": 12}
frame = "ocean"
show_lakes = False
window_size = (3200, 2000)
camera_position = \
    [(-3282418.0538254846, -4079279.745528131, 2942871.552600236),
     (0.0, 0.0, 0.0),
     (0.21481878265999022, 0.4517997331693586, 0.8658694426555174)]
position_txt = (150.0, 680.0)

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
# if plot_sel_dom:
#     plt.figure()
#     ax = plt.axes(projection=crs_rot)
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS)
#     for ind, i in enumerate(list(rot_coords.keys())):
#         rlon_edge = rot_coords[i]["rlon_edge"]
#         rlat_edge = rot_coords[i]["rlat_edge"]
#         poly = [(rlon_edge[0], rlat_edge[0]), (rlon_edge[-1], rlat_edge[0]),
#                 (rlon_edge[-1], rlat_edge[-1]), (rlon_edge[0], rlat_edge[-1])]
#         polygon = plt.Polygon(poly, facecolor="none",
#                               edgecolor=line_prop[ind]["color"],
#                               linewidth=line_prop[ind]["lw"],
#                               linestyle=line_prop[ind]["ls"],)
#         ax.add_patch(polygon)
#     ax.set_extent([rlon_edge[0] - 0.5, rlon_edge[-1] + 0.5,
#                    rlat_edge[0] - 0.5, rlat_edge[-1] + 0.5])

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
    if not show_lakes:
        mask_lake = np.zeros(elevation_in.shape, dtype=bool)
    else:
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
    # visualised elevation -> clip values below sea level to 0.0 m
    # (-> 'cell_data', which is used for colouring, uses unmodified 'elevation')

    # Compute vertices for grid cell columns
    vertices = terrain3d.rect_columns.get_vertices(x_ver, y_ver,
                                                   elevation_pad_0)
    shp_ver = vertices.shape
    vertices_rshp = vertices.reshape((y_ver.size * x_ver.size * 4), 3)

    # Compute quads for grid cell columns
    quads, cell_data, column_index \
        = terrain3d.rect_columns.get_quads(elevation, elevation_pad_0, shp_ver)

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
    grid.cell_data["Surface elevation [m]"] = cell_data_sel

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
clim = (cell_data_min, cell_data_max)
cmap = terrain3d.auxiliary.terrain_colormap(np.array(clim))
color_lake = cm.bukavu(0.3)

# Plot
pos = ((0, 1), (1, 1), (1, 3))
groups = [(0, slice(1, 3)), (1, slice(0, 2)), (1, slice(2, 4))]
pl = pv.Plotter(window_size=window_size, shape=(2, 4), groups=groups)
for ind, i in enumerate(list(rot_coords.keys())):
    pl.subplot(*pos[ind])
    pl.add_mesh(data[i]["grid"], cmap=cmap, clim=clim, show_edges=False,
                show_scalar_bar=False)
    if np.any(data[i]["mask_lake"]):
        pl.add_mesh(data[i]["grid_lake"], color=color_lake, show_edges=False)
    if frame in ("monochrome", "ocean"):
        pl.add_mesh(data[i]["grid_low"], color="lightgrey", show_edges=False)
    txt = "~" + i[:-2] + " " + i[-2:]
    pl.add_text(txt, font_size=25, color="white", position=position_txt)
    pl.set_background("black")
pl.link_views()
pl.camera_position = camera_position
pl.show()
# pl.camera_position  # return camera position when plot is closed
