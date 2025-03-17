# Stream And Scan (SAS)

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Dependencies](https://img.shields.io/badge/dependencies-NumPy-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-Numba-orange)

A Python implementation of the ['Fast Software for Box Intersections'](https://dl.acm.org/doi/10.1145/336154.336192) algorithm  by Zomorodian &amp; Edelsbrunner (2000).

Takes much inspiration from the [Rust port](https://github.com/derivator/box_intersect_ze/tree/main) by "derivator", which itself is based on the [C++ implementation](https://github.com/CGAL/cgal/tree/master/Box_intersection_d/include/CGAL) in CGAL.

## Description
The core algorithm uses a hybrid approach, combining a streamed segment tree with 
scanning techniques, switching between them based on a configurable cutoff threshold.

For lack of a better name, this algorithm will be referred to as "Stream And Scan" (SAS). Unlike data structures such as R-Trees, Quad-Trees, or kd-Trees, this Segment Tree is **streamed** on the fly and never explicitly stored before **scanning**. The name also directly references SAP (Sweep and Prune).

This version contains several adaptations:
- Designed for the "complete case" (self-intersection within one set)
- Simplified median approximation mechanism
- Pre-allocated arrays for storing results
- Numba JIT compilation for performance
- Parallelized operations

Note: the entry function can be easily extended to handle the "bipartite case" (intersection 
between two different sets of boxes) with minimal modifications.

## Example

The main `query_pairs` method takes a single vectorized numpy array containing all AABBs (Axis-Aligned Bounding Boxes). Each AABB is represented as `[min_x, max_x, min_y, max_y]` for a 2d bounding rectangle or as `[min_x, max_x, min_y, max_y, min_z, max_z]` for a 3d bounding box. The algorithm then operates on the entire dataset at once, avoiding individual checks:

```python
# Bulk extraction of all bounding volumes
AABBs = bodies.get_AABB()

# Perform batch query to find all unique colliding pairs
pairs = SAS.query_pairs(AABBs, cutoff=3000)
```
The `cutoff` parameter controls when the algorithm switches from recursive partitioning to direct scanning. Higher values generally yield better performance by minimizing recursion overhead. Default value is `1500`.

## Test
Can maintain 60 fps up to 68,000 unique pairs of colliding rectangles from a set of **20,000** moving AABBs.

*Please note that the above test was carried out without displaying any primitives and only approximately reflects the performance of pure numerical collision detection calculations.
Rendering all bounding rectangles would logically only lower the frame rate.*

Tested on a 2020 MSI laptop with Intel Core i7-8750H CPU @ 2.20GHz and 32Gb RAM
  
## Dependancies

##### Visualization:
- py5 – Main graphical environment for testing and visualization.  
- render_utils (included) – Fast primitive rendering in Py5.  

##### Core dependencies:
- NumPy 
- Numba 
