# Description: Visualise MERIT data for a subregion in Switzerland with
#              a triangle mesh on a planar map projection. Illustrate the
#              algorithm to compute terrain horizon (according to HORAYZON;
#              https://doi.org/10.5194/gmd-15-6817-2022) in an animation.
#              Combining the invividual images of the animation into a movie
#              or GIF requires 'FFmpeg' or 'ImageMagick', respecitvely.
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
import glob
from PIL import Image
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
path_out = os.getenv("HOME") + "/Desktop/PyVista_terrain_horizon/"
# output directory


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
quad_indices = terrain3d.tri_mesh.get_quad_indices(len(lon_ver), len(lat_ver))

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

# Observer (-> location for which horizon is computed)
ind_0 = int(x.shape[0] / 2)
ind_1 = int(x.shape[1] / 2)
observer = np.array([x[ind_0, ind_1], y[ind_0, ind_1],
                     z[ind_0, ind_1] + 1.0])

# Compute terrain horizon (-- perform ray tracing)
data, hori_loc, hori_elev, hori_dist \
    = terrain_horizon(vertices, quad_indices, observer, hori_acc,
                      azim_num, dist_search)
print("Number of steps: " + str(len(data)))

# -----------------------------------------------------------------------------
# Create images
# -----------------------------------------------------------------------------

# Colormap
cmap = terrain3d.auxiliary.terrain_colormap(elevation_ver)

# 3D-images with terrain
camera_position = \
    [(-25430.556484989338, -53353.24821086885, 46453.6498944145),
     (-6.002665031701326e-11, 9.897414867342377, 3815.9759521484375),
     (0.12262671008113055, 0.5831885581520082, 0.8030278921776381)]
for i in list(data.keys()):
    # -------------------------------------------------------------------------
    pl = pv.Plotter(window_size=[3000, 2500], off_screen=True)
    pl.add_mesh(grid, scalars="Surface elevation", show_edges=False, cmap=cmap)
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
    # pl.show()  # set 'off_screen=False' in 'pv.Plotter()' to show plot
    pl.screenshot(path_out + "fig_" + "%03d" % (i + 1) + ".png")
    # pl.camera_position  # return camera position when plot is closed
    pl.close()

# Inset images with horizon angle and distance (optional)
azim = np.arange(0.0, 360.0, 360.0 / azim_num)
hori_elev_real = np.arctan(np.tan(hori_elev) / ter_exa_fac)
num_hori = [data[i]["num_hori"] for i in range(len(data))]
os.mkdir(path_out + "inset_images/")
for i in list(data.keys()):
    fig = plt.figure(figsize=(6.45, 2.5))
    fig.patch.set_facecolor("black")
    # -------------------------------------------------------------------------
    ax0 = plt.axes()
    ax0.set_facecolor("black")
    ax0.tick_params(axis="x", colors="lightgrey")
    ax0.tick_params(axis="y", colors="lightgrey")
    plt.plot(azim[:num_hori[i]], hori_dist[:num_hori[i]] / 1000.0,
             lw=1.0, color="royalblue")
    plt.scatter(azim[:num_hori[i]], hori_dist[:num_hori[i]] / 1000.0,
                s=40, color="royalblue")
    plt.axis([0.0, 345.0, 0.0, dist_search / 1000.0])
    plt.xlabel("Azimuth angle (clockwise from North) [degree]",
               color="lightgrey")
    plt.ylabel("Distance to horizon [km]", labelpad=10)
    ax0.yaxis.label.set_color("royalblue")
    # -------------------------------------------------------------------------
    ax1 = ax0.twinx()
    ax1.set_facecolor("black")
    ax1.tick_params(axis="y", colors="lightgrey")
    ax1.spines["bottom"].set_color("lightgrey")
    ax1.spines["top"].set_color("lightgrey")
    ax1.spines["right"].set_color("lightgrey")
    ax1.spines["left"].set_color("lightgrey")
    plt.plot(azim[:num_hori[i]], np.rad2deg(hori_elev_real)[:num_hori[i]],
             lw=1.5, color="#EC5800")
    plt.scatter(azim[:num_hori[i]], np.rad2deg(hori_elev_real)[:num_hori[i]],
                s=70, color="#EC5800")
    plt.axis([0.0, 345.0, 0.0, np.rad2deg(hori_elev_real).max() * 1.1])
    plt.ylabel("Elevation angle [degree]", labelpad=10)
    ax1.yaxis.label.set_color("#EC5800")
    # -------------------------------------------------------------------------
    fig.savefig(path_out + "inset_images/fig_" + "%03d" % (i + 1) + ".png",
                dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Combine images and create animation (movie or GIF)
# -----------------------------------------------------------------------------

# Merge (and crop) images
images = sorted(glob.glob(path_out + "fig_???.png"))
images_inset = sorted(glob.glob(path_out + "inset_images/fig_???.png"))
if (len(images) != len(images_inset) != len(data)):
    raise ValueError("Images incomplete")
os.mkdir(path_out + "merged/")
box = (200, 600, 3000, 2500)  # (left, upper, right, lower)
for i, j in zip(images, images_inset):
    im = Image.open(i)
    im = im.crop(box)
    im_inset = Image.open(j)
    # im_inset = im_inset.resize((int(im_inset.width * 0.5),
    #                             int(im_inset.height * 0.5)),
    #                            Image.Resampling.LANCZOS)
    # -> scaling can be avoided by setting appropriate dpi for inset images
    im.paste(im_inset, (im.width - im_inset.width,
                        im.height - im_inset.height))
    im.save(path_out + "merged/" + i.split("/")[-1], quality=100)

# Create movie from images (requires 'FFmpeg')
# ffmpeg -framerate 3 -i fig_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p
# terrain_horizon.mp4
# slow down movie: '-framerate 2', accelerate movie: '-framerate 4'

# Create GIF from images (requires 'ImageMagick')
# convert -delay 50 -loop 0 fig_???.png terrain_horizon.gif  # slow
# mogrify -layers 'optimize' -fuzz 7% terrain_horizon.gif    # compress (slow)
