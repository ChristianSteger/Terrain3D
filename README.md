# Terrain3D
Three-dimensional visualisation of terrain data from digital elevation models (DEMs) or climate model topography with [PyVista](https://docs.pyvista.org). A set of Python example scripts illustrates how this data can be plotted for various domains and with auxiliary information.

# Package dependencies

The following Python packages are required to run Terrain3D: NumPy, ...

# Installation

Terrain3D can be installed via:

# Visualisation

A number of examples scripts are provided in the folder *visualisation*:

- **globus.py**: Visualise entire GEBCO data set on sphere with a triangle mesh. The elevation of quad vertices, which are below sea level and are land according to the GSHHG data base, are set to 0.0 m. Ice covered quads (land glaciers or ice shelves) are represented as white areas.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_Globus.png?raw=true "Output from globus.py")

- **switzerland_sub_grid_boxes.py**: Visualise GEBCO or MERIT data for subregion in Switzerland with a triangle mesh. Use a planar map projection and display 'idealised' grid boxes of a GCM/RCM. Optionally display lakes.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/Terrain3D_Switzerland_sub_grid_boxes.png?raw=true "Output from switzerland_sub_grid_boxes.py")