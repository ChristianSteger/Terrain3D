# Terrain3D
Three-dimensional visualisation of terrain data from digital elevation models (DEMs) or climate model topography with [PyVista](https://docs.pyvista.org). A set of Python example scripts illustrates how this data can be plotted for various domains and with auxiliary information.

When you use Terrain3D, please cite:

[![DOI](https://zenodo.org/badge/599311358.svg)](https://zenodo.org/badge/latestdoi/599311358)

# Package dependencies

The following Python packages are required to run Terrain3D: NumPy, SciPy, Xarray, Numba, Shapely, descartes, Fiona, Rasterio, tqdm, requests, PyVista, Matplotlib, pyproj, cmcrameri, netCDF4, cartopy, xESMF, scikit-image and Pillow.
Combining the individual output images of the example scripts **tri_mesh_globe.py** and **triangles_terrain_horizon.py** into a movie or GIF requires [FFmpeg](https://ffmpeg.org) or [ImageMagick](https://imagemagick.org/index.php), respecitvely.

# Installation

First, create a Conda environment with all the required Python packages:

```bash
conda create -n terrain3d -c conda-forge numpy scipy matplotlib netcdf4 shapely xarray pyproj cartopy rasterio descartes fiona scikit-image numba xesmf cmcrameri tqdm requests pyvista pillow
```

and **activate this environment**. The Terrain3D package can then be installed with:

```bash
git clone https://github.com/ChristianSteger/Terrain3D.git
cd Terrain3D
python -m pip install .
```

**Known issues**

Under **Mac OS X**, the current default version of xESMF that is installed with Conda might cause problems. This can be resolved by installed a specific version (xesmf=0.7.0) and adding

```bash
import os
os.environ["ESMFMKFILE"] = "<specify path to file 'esmf.mk'; something like ../miniconda3/envs/terrain3d/lib/esmf.mk>"
import xesmf as xe
```

to visualisation scripts, in which the xESMF library is imported.

# Visualisation

A number of examples scripts are provided in the folder *visualisation*:

- **tri_mesh_globe.py**: Visualise entire GEBCO data set on sphere with a triangle mesh. The elevation of quad vertices, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Ice covered quads (land glaciers or ice shelves) are represented as white areas.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Tri_mesh_globe.png?raw=true "Output from tri_mesh_globe.py")

- **tri_mesh_vertical_grid.py**: Visualise GEBCO or MERIT data for subregion in Switzerland with a triangle mesh. Use a planar map projection and display 'idealised' vertical grid of a GCM/RCM. Optionally display lakes.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Tri_mesh_vertical_grid.png?raw=true "Output from tri_mesh_vertical_grid.py")

- **tri_mesh_terrain_horizon.py**: Visualise MERIT data for a subregion in Switzerland with a triangle mesh on a planar map projection. Illustrate the algorithm to compute terrain horizon (according to [HORAYZON](https://doi.org/10.5194/gmd-15-6817-2022)) in an animation. Combining the invividual images of the animation into a movie or GIF requires [FFmpeg](https://ffmpeg.org) or [ImageMagick](https://imagemagick.org/index.php), respecitvely.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Tri_mesh_terrain_horizon.gif?raw=true "Output from tri_mesh_terrain_horizon.py")

- **tri_mesh_cosmo_multi.py**: Visualise COSMO topography for the Hengduan Mountains (Southeastern Tibetan Plateau) with a triangle mesh. Plot three different topographies (present-day, reduced and envelope).
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Tri_mesh_cosmo_multi.png?raw=true "Output from tri_mesh_cosmo_multi.py")

- **tri_mesh_aster_icon_radar.py**: Visualise raw ASTER 30 m DEM and ICON 1km topography with radar location.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Tri_mesh_aster_icon_radar.png?raw=true "Output from rect_columns_cosmo_vertical_grid.py")

- **rect_columns_gebco.py**: Visualise GEBCO data set with rectangular columns (&rarr; terrain representation in climate models). The elevation of grid cells, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Lakes can optionally be displayed.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Rect_columns_gebco.png?raw=true "Output from rect_columns_gebco.py")

- **rect_columns_gebco_res_ch.py**: Visualise GEBCO data set for Switzerland with rectangular columns (&rarr; terrain representation in climate models). The elevation of grid cells, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Lakes are additionally displayed. Different spatial resolutions are visualised.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Rect_columns_gebco_res_ch.png?raw=true "Output from rect_columns_gebco_res_ch.py")

- **rect_columns_gebco_res_eu.py**: Visualise GEBCO data set for Middle/South Europe with rectangular columns (&rarr; terrain representation in climate models). The elevation of grid cells, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Different spatial resolutions are visualised.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Rect_columns_gebco_res_eu.png?raw=true "Output from rect_columns_gebco_res_eu.py")

- **rect_columns_cosmo_vertical_grid.py**: Visualise COSMO topography for a subregion of the Alps with rectangular columns. Vertical height-based hybrid (Gal-Chen) coordinates are additionally represented.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D/Rect_columns_cosmo_vertical_grid.png?raw=true "Output from rect_columns_cosmo_vertical_grid.py")

# Digital elevation model and auxiliary data

The following DEM data is available in Terrain3D:

- [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)
- [MERIT](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)

And the following auxiliary data is applied:

- [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)
- [Natural Earth - Glaciated Areas](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-glaciated-areas/)
- [Natural Earth - Antarctic Ice Shelves](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-antarctic-ice-shelves/)
- [swissTLMRegio - Swiss lake outlines](https://www.swisstopo.admin.ch/de/landschaftsmodell-swisstlmregio)

# Support and collaboration

In case of issues or questions, contact Christian R. Steger (christian.steger@env.ethz.ch). Please report any bugs you find in Terrain3D. You are welcome to fork this repository to modify the source code.