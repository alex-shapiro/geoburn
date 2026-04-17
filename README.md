# Geoburn

This repo is an experimental geometry kernel.
It uses nalgebra as its linear algebraic base.

Use it at your own risk.

## Layers

From least to most abstract:

- **gp** is the lowest-level geometry processing module
- **curves** and **surfaces** are low-level 1D and 2D utilities
- **bnd** is bounding box logic
- **bspline** contains b-spline curve and surface logic
- **geom** contains 2/3D curve and surface definitions
- **shape** contains 2/3D shape definitions
- **solids** contains builder functions for solids

## TODO

- https://github.com/thingi10k/thingi10k
