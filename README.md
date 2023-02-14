# Terrain3D
Three-dimensional visualisation of terrain data from digital elevation models (DEMs) or climate model topography with [PyVista](https://docs.pyvista.org). A set of Python example scripts illustrates how this data can be plotted for various domains and with auxiliary information.

# Package dependencies

The following Python packages are required to run Terrain3D: NumPy, SciPy, Xarray, Numba, Shapely, descartes, Fiona, Rasterio, tqdm, requests, PyVista, Matplotlib, pyproj, cmcrameri, netCDF4, cartopy, xESMF, scikit-image

# Installation

First, create a Conda environment with all the required Python packages:

```bash
conda create -n terrain3d -c conda-forge numpy scipy matplotlib netcdf4 shapely xarray pyproj cartopy rasterio descartes fiona scikit-image numba xesmf cmcrameri tqdm requests pyvista
```

and **activate this environment**. The Terrain3D package can then be installed with:

```bash
git clone https://github.com/ChristianSteger/Terrain3D.git
cd Terrain3D
python -m pip install .
```

**Known issues**

Under Mac OS X, the current default version of xESMF that is installed with Conda might cause problems. This can be resolved by installed a specific version (xesmf=0.7.0) and adding

```bash
os.environ["ESMFMKFILE"] = "<specify path to file 'esmf.mk'; something like ../miniconda3/envs/terrain3d/lib/esmf.mk>"
import xesmf as xe
```

to visualisation scripts, in which the xESMF library is imported.

# Visualisation

A number of examples scripts are provided in the folder *visualisation*:

- **triangles_globe.py**: Visualise entire GEBCO data set on sphere with a triangle mesh. The elevation of quad vertices, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Ice covered quads (land glaciers or ice shelves) are represented as white areas.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_Globus.png?raw=true "Output from triangles_globe.py")

- **triangles_grid_boxes.py**: Visualise GEBCO or MERIT data for subregion in Switzerland with a triangle mesh. Use a planar map projection and display 'idealised' grid boxes of a GCM/RCM. Optionally display lakes.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_Switzerland_sub_grid_boxes.png?raw=true "Output from triangles_grid_boxes.py")

- **columns_rot_coords_gebco.py**: Visualise GEBCO data set with 'grid cell columns' (-> terrain representation in climate models). The elevation of grid cells, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Lakes can optionally be displayed.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_columns_rot_coords_gebco.png?raw=true "Output from columns_rot_coords_gebco.py")


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