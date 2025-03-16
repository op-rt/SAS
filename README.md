# Stream And Scan (SAS)

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Dependencies](https://img.shields.io/badge/dependencies-NumPy-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-Numba-orange)


A Python implementation of the ['Fast Software for Box Intersections'](https://dl.acm.org/doi/10.1145/336154.336192) algorithm  by Zomorodian &amp; Edelsbrunner (2000)

The core algorithm uses a hybrid approach, combining a streamed segment tree with 
scanning techniques, switching between them based on a configurable cutoff threshold.

(For lack of a name this algorithm will be referred to as "SAS" -Stream And Scan-)

This version contains several adaptations:
- Simplified median approximation mechanism
- Pre-allocated arrays for storing results
- Numba JIT compilation for performance
- Parallelized operations

Takes much inspiration from the [Rust port](https://github.com/derivator/box_intersect_ze/tree/main) by "Derivator"

Which itself is based on the [C++ implementation](https://github.com/CGAL/cgal/tree/master/Box_intersection_d/include/CGAL) in CGAL
