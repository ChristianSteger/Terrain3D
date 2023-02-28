# Description: Visualise MERIT data for subregion in Switzerland with
#              a triangle mesh. Use a planar map projection and visualise
#              computation of terrain horizon according to the HORAYZON
#              algorithm.
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import vtk
import pyvista as pv
from cmcrameri import cm
from pyproj import CRS
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib as mpl
import subprocess
import glob
import terrain3d

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

ter_exa_fac = 1.5  # terrain exaggeration factor [-]
domain = ( 9.231788 - 0.2, 9.231788 + 0.2,
          47.119357 - 0.15, 47.119357 + 0.15)  # Walensee (Switzerland)
dist_search = 10000.0  # search distance for terrain horizon [m]
hori_acc = np.deg2rad(1.0)  # accuracy of horizon computation [degree]
azim_num = 24  # number of azimuth sectors [-]
path_out = "/Users/csteger/Desktop/3D_images/"  # output directory


# -----------------------------------------------------------------------------
# Function to compute terrain horizon and output intermediate steps
# -----------------------------------------------------------------------------

def terrain_horizon(vertices, quad_indices, observer, hori_acc, azim_num,
                    dist_search):

    # Create mesh for terrain (poly data -> required for ray tracing)
    mesh = pv.PolyData(vertices, quad_indices.ravel())

    # Data structures for output
    data = {}  # output data per step
    hori_loc = np.empty((azim_num, 3), dtype=np.float32)
    hori_elev = np.empty(azim_num, dtype=np.float32)
    hori_dist = np.empty(azim_num, dtype=np.float32)

    # Initialisation
    elev = np.deg2rad(0.0)
    azim = np.deg2rad(0.0)
    step = 0
    num_hori = 0

    # -------------------------------------------------------------------------
    # First azimuth direction (discrete sampling)
    # -------------------------------------------------------------------------

    flag_continue = True
    while flag_continue:

        # Perform ray tracing
        x_ray = dist_search * np.cos(elev) * np.sin(azim)
        y_ray = dist_search * np.cos(elev) * np.cos(azim)
        z_ray = dist_search * np.sin(elev)
        ray_end = observer + np.array([x_ray, y_ray, z_ray])
        points, ind = mesh.ray_trace(observer, ray_end)
        ray = pv.Line(observer, ray_end)
        if points.shape[0] >= 1:
            terrain_inters = points[0, :]
            dist_inters = np.sqrt(((terrain_inters - observer) ** 2).sum())
            elev += hori_acc * 2.0
        else:
            terrain_inters = None
            elev = elev - hori_acc
            x_ray = dist_inters * np.cos(elev) * np.sin(azim)
            y_ray = dist_inters * np.cos(elev) * np.cos(azim)
            z_ray = dist_inters * np.sin(elev)
            hori_loc[num_hori, :] = observer + np.array([x_ray, y_ray, z_ray])
            hori_elev[num_hori] = elev
            hori_dist[num_hori] = dist_inters
            dist_inters = np.nan
            num_hori += 1
            flag_continue = False

        data[step] = {"ray": ray, "terrain_inters": terrain_inters,
                      "num_hori": num_hori}
        step += 1

    # -------------------------------------------------------------------------
    # Remaining azimuth directions (guess horizon from previous azimuth
    # direction)
    # -------------------------------------------------------------------------

    for i in range(1, azim_num):

        azim += (2.0 * np.pi) / azim_num

        # Perform initial ray tracing for azimuth direction
        x_ray = dist_search * np.cos(elev) * np.sin(azim)
        y_ray = dist_search * np.cos(elev) * np.cos(azim)
        z_ray = dist_search * np.sin(elev)
        ray_end = observer + np.array([x_ray, y_ray, z_ray])
        points, ind = mesh.ray_trace(observer, ray_end)
        ray = pv.Line(observer, ray_end)
        if points.shape[0] == 0:  # no hit
            elev_delta = -(hori_acc * 2.0)  # move downwards
            terrain_inters = None
            dist_inters = np.nan
        else:
            elev_delta = (hori_acc * 2.0)  # move upwards
            terrain_inters = points[0, :]
            dist_inters = np.sqrt(((terrain_inters - observer) ** 2).sum())
        data[step] = {"ray": ray, "terrain_inters": terrain_inters,
                      "num_hori": num_hori}
        step += 1

        # Move up- or downwards
        flag_continue = True
        while flag_continue:

            # Perform ray tracing
            elev_prev = elev
            elev += elev_delta
            x_ray = dist_search * np.cos(elev) * np.sin(azim)
            y_ray = dist_search * np.cos(elev) * np.cos(azim)
            z_ray = dist_search * np.sin(elev)
            ray_end = observer + np.array([x_ray, y_ray, z_ray])
            points, ind = mesh.ray_trace(observer, ray_end)
            ray = pv.Line(observer, ray_end)
            dist_inters_prev = dist_inters
            if points.shape[0] >= 1:
                terrain_inters = points[0, :]
                dist_inters = np.sqrt(((terrain_inters - observer) ** 2).sum())
            else:
                terrain_inters = None
                dist_inters = np.nan

            # Check for break condition
            if (((elev_delta > 0.0) and (terrain_inters is  None))
                    or ((elev_delta < 0.0) and (terrain_inters is not None))):
                dist_inters = np.nanmean([dist_inters_prev, dist_inters])
                elev = np.mean([elev_prev, elev])
                x_ray = dist_inters * np.cos(elev) * np.sin(azim)
                y_ray = dist_inters * np.cos(elev) * np.cos(azim)
                z_ray = dist_inters * np.sin(elev)
                hori_loc[num_hori, :] \
                    = observer + np.array([x_ray, y_ray, z_ray])
                hori_elev[num_hori] = elev
                hori_dist[num_hori] = dist_inters
                num_hori += 1
                flag_continue = False

            data[step] = {"ray": ray, "terrain_inters": terrain_inters,
                          "num_hori": num_hori}
            step += 1

    return data, hori_loc, hori_elev, hori_dist

# -----------------------------------------------------------------------------
# Prepare data
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise ValueError("Output directory does not exist")

# Get MERIT data
lon_ver, lat_ver, elevation_ver, crs_dem = terrain3d.merit.get(domain)

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

# Create mesh for terrain (unstructured grid)
cell_types = np.empty(quad_indices.shape[0], dtype=np.uint8)
cell_types[:] = vtk.VTK_QUAD
grid = pv.UnstructuredGrid(quad_indices.ravel(), cell_types, vertices)
grid.point_data["Surface elevation"] = elevation_ver.ravel()

# -----------------------------------------------------------------------------
# Perform ray tracing and visualise steps
# -----------------------------------------------------------------------------

# Observer (-> location for which horizon is computed)
ind_0 = int(x.shape[0] / 2)
ind_1 = int(x.shape[1] / 2)
observer = np.array([x[ind_0, ind_1], y[ind_0, ind_1],
                     z[ind_0, ind_1] + 1.0])

# Compute terrain horizon
data, hori_loc, hori_elev, hori_dist \
    = terrain_horizon(vertices, quad_indices, observer, hori_acc,
                      azim_num, dist_search)
print("Number of steps: " + str(len(data)))

# Plot horizon (angle and distance)
azim = np.arange(0.0, 360.0, 360.0 / azim_num)
hori_elev_real = np.arctan(np.tan(hori_elev) / ter_exa_fac)
fig = plt.figure(figsize=(16, 5))
ax0 = plt.axes()
plt.fill_between(x=azim, y1=0.0, y2=np.rad2deg(hori_elev_real),
                 color="lightgrey")
plt.plot(azim, np.rad2deg(hori_elev_real), lw=1.5, color="orangered")
plt.scatter(azim, np.rad2deg(hori_elev_real), s=70, color="orangered")
plt.ylim([0.0, 34.0])
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
ax0.yaxis.label.set_color("orangered")
ax1 = ax0.twinx()
plt.plot(azim, hori_dist / 1000.0, lw=1.5, color="royalblue")
plt.xlim([0.0, 345.0])
plt.ylabel("Distance to horizon [km]")
ax1.yaxis.label.set_color("royalblue")
fig.savefig(path_out + "Horizon_angle_and_distance.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# Plot steps
colormap = terrain3d.auxiliary.cmap_terrain(elevation_ver, cm.bukavu)
camera_position = \
    [(-25430.556484989338, -53353.24821086885, 46453.6498944145),
     (-6.002665031701326e-11, 9.897414867342377, 3815.9759521484375),
     (0.12262671008113055, 0.5831885581520082, 0.8030278921776381)]
for i in list(data.keys()):
    # -------------------------------------------------------------------------
    pl = pv.Plotter(window_size=[2500, 2500])
    pl.add_mesh(grid, scalars="Surface elevation", show_edges=False,
                edge_color="black", line_width=0, cmap=colormap)
    # -------------------------------------------------------------------------
    pl.add_mesh(data[i]["ray"], color="black", line_width=6)
    pt_obs = pv.Sphere(radius=250, center=observer)
    pl.add_mesh(pt_obs, color="#87CEFA", show_edges=False)
    # -------------------------------------------------------------------------
    if data[i]["terrain_inters"] is not None:
        pt_inters = pv.Sphere(radius=250, center=data[i]["terrain_inters"])
        pl.add_mesh(pt_inters, color="black", show_edges=False)
    # -------------------------------------------------------------------------
    for j in range(0, data[i]["num_hori"]):
        sphere_hori = pv.Sphere(radius=250, center=hori_loc[j, :])
        pl.add_mesh(sphere_hori, color="#EC5800", show_edges=False)
    # -------------------------------------------------------------------------
    pl.remove_scalar_bar()
    pl.set_background("black")
    pl.camera_position = camera_position
    # pl.show()
    pl.show(screenshot=path_out + "fig_" + "%03d" % (i + 1) + ".png")
    # pl.camera_position  # return camera position when plot is closed
    pl.close()

# Crop images (optional)
files = glob.glob(path_out + "fig_???.png")
files.sort()
print("Number of files: " + str(len(files)))
if not os.path.isdir(path_out + "cropped/"):
    os.mkdir(path_out + "cropped/")
for i in files:
    cmd = ("convert", "-crop 2500x1900+0+600")
    sf = i
    tf = path_out + "cropped/" + i.split("/")[-1]
    subprocess.call(cmd[0] + " " + sf + " " + cmd[1] + " " + tf, shell = True)
    # os.remove(sf)

# Create movie from images:
# ffmpeg -framerate 2 -i fig_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out2.mp4  # slowest
# ffmpeg -framerate 3 -i fig_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out3.mp4
# ffmpeg -framerate 4 -i fig_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out4.mp4  # fastest