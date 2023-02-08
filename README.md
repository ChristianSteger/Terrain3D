# Terrain3D
Three-dimensional visualisation of terrain data from digital elevation models (DEMs) or climate model topography with [PyVista](https://docs.pyvista.org). A set of Python example scripts illustrates how this data can be plotted for various domains and with auxiliary information.

# Package dependencies

The following Python packages are required to run Terrain3D: NumPy, SciPy, Xarray, Numba, Shapely, Fiona, Rasterio, tqdm, requests, PyVista, Matplotlib, pyproj, cmcrameri

# Installation

Terrain3D can be installed via:

# Visualisation

A number of examples scripts are provided in the folder *visualisation*:

- **globus.py**: Visualise entire GEBCO data set on sphere with a triangle mesh. The elevation of quad vertices, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Ice covered quads (land glaciers or ice shelves) are represented as white areas.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_Globus.png?raw=true "Output from globus.py")

- **switzerland_sub_grid_boxes.py**: Visualise GEBCO or MERIT data for subregion in Switzerland with a triangle mesh. Use a planar map projection and display 'idealised' grid boxes of a GCM/RCM. Optionally display lakes.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_Switzerland_sub_grid_boxes.png?raw=true "Output from switzerland_sub_grid_boxes.py")

# Digital elevation model and auxiliary data

The following DEM data is available in Terrain3D:

- [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)
- [MERIT](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)

And the following auxiliary data is applied:

- [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)
- [Natural Earth - Glaciated Areas](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-glaciated-areas/)
- [Natural Earth - Antarctic Ice Shelves](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-antarctic-ice-shelves/)

# Support and collaboration

In case of issues or questions, contact Christian R. Steger (christian.steger@env.ethz.ch). Please report any bugs you find in Terrain3D. You are welcome to fork this repository to modify the source code.