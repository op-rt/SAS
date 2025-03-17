"""
Efficient Broad Phase Collision Detection of Boxes (3d/2d)

An implementation of the 'Fast Software for Box Intersections' algorithm 
by Zomorodian & Edelsbrunner (2000)
(https://dl.acm.org/doi/10.1145/336154.336192)

The core algorithm uses a hybrid approach, combining a streamed segment tree with 
scanning techniques, switching between them based on a configurable cutoff threshold.

(For lack of a name this algorithm will be referred to as "SAS" -Stream And Scan-)

This version contains several adaptations:
- Designed for the "complete case" (self-intersection within one set)
- Simplified median approximation mechanism
- Pre-allocated arrays for storing results
- Numba JIT compilation for performance
- Parallelized operations

Implementation insights:
- Strict inequalities for true overlaps decreases the number of colliding pairs.
- High cutoff values (10000+) deliver best performance by avoiding recursion overhead.
- Delivers marginally better performance than SAP for large datasets (30,000+ boxes).
- A non-recursive stack-based approach would potentially have better memory locality.

Takes much inspiration from the Rust port by "Derivator" 
(https://github.com/derivator/box_intersect_ze/tree/main)

Which itself is based on the C++ implementation in CGAL
(https://github.com/CGAL/cgal/tree/master/Box_intersection_d/include/CGAL)


# Author: Louis D. 
# Created: 2023-03-04
# Python Version: 3.11
# Context: Need for a robust algorithm to detect broad-phase collisions between 2 or 
#          3-dimensional AABBs, reducing costly narrow-phase calculations in a Dynamic 
#          Relaxation solver.

"""

from numba import njit, prange, config
import numpy as np

num_threads = config.NUMBA_NUM_THREADS

@njit("int32(int32[:, :], int32[:], int32[:, :], int32[:], int32, int32[:, :], int32)", nogil=True, cache=True, fastmath=True, parallel=True)
def one_way_scan(intervals, interval_ids, points, point_ids, max_dim_check, pairs, count):

    """
    Reports intersections between intervals and points by scanning in dimension 0,
    treating boxes in points as points: intersections are only reported when the low
    endpoint in dimension 0 of a box in points is inside the projection of a box in intervals.
    
    Parameters
    ----------
    intervals     (2d array): First set of AABBs treated as intervals
    interval_ids  (1d array): Array of indices for intervals
    points        (2d array): Second set of AABBs treated as points
    point_ids     (1d array): Array of indices for points
    max_dim_check (int):      Highest dimension to check for intersection
    pairs         (2d array): Pre-allocated array to store intersection results
    count         (int):      Current count of intersections found
    
    Returns
    -------
    count         (int):      Updated count of intersections found

    """

    p_len = points.shape[0]
    n_intervals = intervals.shape[0]
    
    # Pre-sort points by their lower bound in dimension 0 if not already sorted
    sort_idx = np.argsort(points[:, 0])
    sorted_points = points[sort_idx]
    sorted_point_ids = point_ids[sort_idx]
    
    # Local results array to avoid race conditions with parallel processing
    # Each thread will store its pairs in a separate section
    local_results = np.empty((n_intervals, p_len, 2), dtype=np.int32)
    local_counts = np.zeros(n_intervals, dtype=np.int32)
    
    # Iterate through intervals in parallel
    for i_idx in prange(n_intervals):
        i = intervals[i_idx]
        i_id = interval_ids[i_idx]
        i_min = i[0]  # lo(0)
        i_max = i[1]  # hi(0)
        
        p_min_idx = 0
        # Skip all points that don't have a chance to be in i
        while p_min_idx < p_len and sorted_points[p_min_idx, 0] < i_min:
            p_min_idx += 1
        
        # If no point has a chance to be in the current interval, skip to next
        if p_min_idx == p_len:
            continue
        
        # Check all potentially intersecting points
        for p_idx in range(p_min_idx, p_len):
            p = sorted_points[p_idx]
            p_id = sorted_point_ids[p_idx]
            p_min = p[0]  # lo(0)
            
            # Early exit if we're past this interval
            if p_min >= i_max:
                break
            
            # Skip self-intersection
            if p_id == i_id:
                continue
            
            # Check intersection in all required dimensions
            intersects = True
            for dim in range(1, max_dim_check + 1):
                dim_low = dim * 2
                dim_high = dim_low + 1
                
                # Check if p intersects i in this dimension
                if not (p[dim_low] < i[dim_high] and p[dim_high] > i[dim_low]):
                    intersects = False
                    break
            
            if not intersects:
                continue
            
            # If low endpoints are equal, use ID comparison to avoid duplicates
            if p_min == i_min and p_id > i_id:
                continue
            
            # Store result in thread-local array
            if local_counts[i_idx] < p_len:
                local_results[i_idx, local_counts[i_idx], 0] = i_id
                local_results[i_idx, local_counts[i_idx], 1] = p_id
                local_counts[i_idx] += 1
    
    # Merge results from all threads
    total_count = 0
    for i_idx in range(n_intervals):
        local_count = local_counts[i_idx]
        # Copy results to output array
        for j in range(local_count):
            if count + total_count < pairs.shape[0]:
                pairs[count + total_count, 0] = local_results[i_idx, j, 0]
                pairs[count + total_count, 1] = local_results[i_idx, j, 1]
                total_count += 1
    
    return count + total_count


@njit("int32(int32[:, :], int32[:], int32[:, :], int32[:], int32, int32[:, :], int32)", nogil=True, cache=True, fastmath=True, parallel=True)
def two_way_scan(intervals, interval_ids, points, point_ids, max_dim_check, pairs, count):

    """
    Parallelized two-way scan with streaming for optimal performance.

    Parameters
    ----------
    intervals     (2d array): First set of AABBs treated as intervals
    interval_ids  (1d array): Array of indices for intervals
    points        (2d array): Second set of AABBs treated as points
    point_ids     (1d array): Array of indices for points
    max_dim_check (int):      Highest dimension to check for intersection
    pairs         (2d array): Pre-allocated array to store intersection results
    count         (int):      Current count of intersections found
    
    Returns
    -------
    count         (int):      Updated count of intersections found

    """
    
    i_len = intervals.shape[0]
    p_len = points.shape[0]
    
    # If one of the inputs is empty, return
    if i_len == 0 or p_len == 0:
        return count
    
    # For parallel processing, divide intervals into chunks
    chunk_size = max(1, min(100, i_len // (num_threads * 2)))
    num_chunks = (i_len + chunk_size - 1) // chunk_size
    
    # Create local storage for each chunk's results
    max_results_per_chunk = min(10000, p_len * 10)  # Reasonable upper bound
    local_results = np.empty((num_chunks, max_results_per_chunk, 2), dtype=np.int32)
    local_counts = np.zeros(num_chunks, dtype=np.int32)
    
    # Process chunks in parallel
    for chunk in prange(num_chunks):
        i_start = chunk * chunk_size
        i_end = min(i_start + chunk_size, i_len)
        
        # Process intervals in this chunk
        for i_idx in range(i_start, i_end):
            interval = intervals[i_idx]
            i_id = interval_ids[i_idx]
            i_min_x = interval[0]
            i_max_x = interval[1]
            
            # Find starting point - skip points whose MAX_x is <= interval's MIN_x
            p_start_idx = 0
            while p_start_idx < p_len and points[p_start_idx, 1] <= i_min_x:
                p_start_idx += 1
            
            # Check points that might intersect
            for p_idx in range(p_start_idx, p_len):
                point = points[p_idx]
                p_id = point_ids[p_idx]
                p_min_x = point[0]
                
                # Early exit - no more intersections possible
                if p_min_x >= i_max_x:
                    break
                
                # Skip self-intersection
                if i_id == p_id:
                    continue
                
                # Check intersection in all dimensions
                intersects = True
                for dim in range(max_dim_check + 1):
                    min_idx = dim * 2
                    max_idx = min_idx + 1
                    
                    if not (interval[min_idx] < point[max_idx] and interval[max_idx] > point[min_idx]):
                        intersects = False
                        break
                
                if not intersects:
                    continue
                
                # Apply the canonical segment filter
                max_dim = max_dim_check
                min_idx = max_dim * 2
                max_idx = min_idx + 1
                
                # Only include if point's min in max dimension is inside interval's range
                if not (interval[min_idx] <= point[min_idx] < interval[max_idx]):
                    continue
                
                # Tie-breaking to avoid duplicates
                if interval[min_idx] == point[min_idx] and i_id > p_id:
                    continue
                
                # Store in local results for this chunk
                local_idx = local_counts[chunk]
                if local_idx < max_results_per_chunk:
                    local_results[chunk, local_idx, 0] = i_id
                    local_results[chunk, local_idx, 1] = p_id
                    local_counts[chunk] += 1
    
    # Merge results from all chunks
    total_new_pairs = 0
    for chunk in range(num_chunks):
        chunk_count = local_counts[chunk]
        for i in range(chunk_count):
            if count + total_new_pairs < pairs.shape[0]:
                pairs[count + total_new_pairs] = local_results[chunk, i]
                total_new_pairs += 1
    
    return count + total_new_pairs



@njit("float32(int32[:, :], int32, int32)", nogil=True, cache=True, fastmath=True)
def approx_median(boxes, dim, k):

    """
    Find an approximate median value for the specified dimension of boxes.
    
    Simplified implementation of the median approximation that extracts values
    from a specific dimension and uses seeded randomness for reproducibility.
    
    Parameters
    ----------
    boxes (2d array): Array of AABBs with min/max coordinates
    dim   (int):      Dimension to find median in
    k     (int):      Height parameter controlling sampling amount
        
    Returns
    -------
    median (float):   Approximate median value as float32

    """

    # Extract the relevant column
    P = boxes[:, dim * 2]
    n = P.shape[0] 
    if n == 0:
        return np.float32(np.inf)

    # Compute sample size - limit to prevent excessive computation
    sample_size = min(3 ** k, n, 500)  # Cap at 500 samples
    
    # Fast path for small arrays - use the whole array
    if n <= sample_size:
        return np.float32(np.median(P))
    
    # Create array for samples
    sampled = np.empty(sample_size, dtype=np.int32)
    
    # Generate random indices and collect samples
    for i in range(sample_size):
        idx = np.random.randint(0, n)
        sampled[i] = P[idx]
    
    # For very small sample sizes, use a faster sorting method
    if sample_size <= 9:
        # Sort in-place and find middle element
        sampled.sort()
        return np.float32(sampled[sample_size // 2])
    else:
        return np.float32(np.median(sampled))


@njit("int32(int32[:, :], int32[:], int32[:, :], int32[:], float32, float32, int32, int32, int32[:, :], int32)", nogil=True, cache=False, fastmath=True)
def hybrid(intervals, interval_ids, points, point_ids, lo, hi, dim, count, pairs, cutoff):
    """
    Main recursive procedure implementing the hybrid box intersection algorithm.
    
    Recursively processes boxes by dimension, using approximate medians for partitioning.
    Switches to scanning when the number of boxes falls below the cutoff threshold.
    
    Parameters
    ----------
    intervals     (2d array): First set of AABBs with min/max coordinates
    interval_ids  (1d array): Array of indices for intervals
    points        (2d array): Second set of AABBs with min/max coordinates
    point_ids     (1d array): Array of indices for points
    lo            (float):    Lower bound of current segment
    hi            (float):    Upper bound of current segment
    dim           (int):      Current dimension to process
    count         (int):      Current count of intersections found
    pairs         (2d array): Pre-allocated array to store intersection results
    cutoff        (int):      Threshold below which to switch to scanning
        
    Returns
    -------
    count         (int):      Updated count of intersections found

    """

    # Step 1: return if input is empty
    if intervals.shape[0] == 0 or points.shape[0] == 0 or hi <= lo:
        return count

    # Step 2: first hybridization method: scan if only dimension 0 is left to check
    if dim == 0:
        return one_way_scan(intervals, interval_ids, points, point_ids, 0, pairs, count)

    # Step 3: second hybridization method: scan if size of input is smaller than cutoff
    if intervals.shape[0] < cutoff or points.shape[0] < cutoff:
        return two_way_scan(intervals, interval_ids, points, point_ids, dim, pairs, count)

    # Step 4: let intervals_m contain the intervals that would be stored at this node of the segment tree
    # because they span the segment [lo, hi), meaning it is one of their canonical segments
    dim_low = dim * 2
    dim_high = dim_low + 1
    
    # A box spans the segment if its low coordinate is below lo and high coordinate is above hi
    mask_spanning = (intervals[:, dim_low] < lo) & (intervals[:, dim_high] > hi)
    
    # Create separate arrays for spanning and non-spanning boxes
    intervals_m = intervals[mask_spanning]  # Spanning intervals stored at this node
    interval_ids_m = interval_ids[mask_spanning]
    
    intervals_lr = intervals[~mask_spanning]  # Non-spanning intervals
    interval_ids_lr = interval_ids[~mask_spanning]

    # Process spanning intervals with TWO bidirectional recursive calls in next dimension
    if intervals_m.shape[0] > 0:
        # First call: spanning intervals vs points
        count = hybrid(intervals_m, interval_ids_m, points, point_ids, np.float32(-np.inf), np.float32(np.inf), dim - 1, count, pairs, cutoff)
        
        # Second call: points vs spanning intervals (reversed)
        count = hybrid(points, point_ids, intervals_m, interval_ids_m, np.float32(-np.inf), np.float32(np.inf), dim - 1, count, pairs, cutoff)

    # If no non-spanning intervals, we're done
    if intervals_lr.shape[0] == 0:
        return count

    # Step 5: divide the segment [lo, hi) into subsegments by approximate median
    k = min(3, int(np.log(points.shape[0] / 1000) / np.log(3)) if points.shape[0] > 1000 else 0)
    mi = approx_median(points, dim, k)

    # If we failed to divide the segment, just scan
    if mi == hi or mi == lo:
        return two_way_scan(intervals_lr, interval_ids_lr, points, point_ids, dim, pairs, count)

    # Partition points by the median
    mask_points_l = points[:, dim_low] < mi
    points_l = points[mask_points_l]
    point_ids_l = point_ids[mask_points_l]
    
    points_r = points[~mask_points_l]
    point_ids_r = point_ids[~mask_points_l]

    # Partition non-spanning intervals by the median
    # intervals_l and intervals_r are not usually disjoint!
    mask_intervals_l = intervals_lr[:, dim_low] < mi
    intervals_l = intervals_lr[mask_intervals_l]
    interval_ids_l = interval_ids_lr[mask_intervals_l]
    
    mask_intervals_r = intervals_lr[:, dim_high] > mi
    intervals_r = intervals_lr[mask_intervals_r]
    interval_ids_r = interval_ids_lr[mask_intervals_r]

    # Step 6 & 7: Recurse on both partitions
    if intervals_l.shape[0] > 0 and points_l.shape[0] > 0:
        count = hybrid(intervals_l, interval_ids_l, points_l, point_ids_l, lo, mi, dim, count, pairs, cutoff)
    
    if intervals_r.shape[0] > 0 and points_r.shape[0] > 0:
        count = hybrid(intervals_r, interval_ids_r, points_r, point_ids_r, mi, hi, dim, count, pairs, cutoff)

    return count


@njit(cache=True)
def query_pairs(boxes, cutoff=1500):

    """
    Entry point function for finding all box intersections.
    
    Prepares input data, configures parameters, and calls the hybrid algorithm.
    Processes the complete case (boxes intersecting themselves).

    Note: In high-density configurations (e.g., AABBs with near-identical coordinates),
    consider slightly spreading out objects to prevent spatial overlap ambiguity, which can
    cause duplicate pairs. This is a known limitation of the current implementation.
    
    Parameters
    ----------
    AABBs  (2d array): Array of axis-aligned bounding boxes
    cutoff (int):      Threshold below which to switch from recursion to scanning (default=1500)
        
    Returns
    -------
    pairs  (2d array): Pairs of indices representing intersecting boxes

    """

    # Determine dimensionality from input shape
    num_boxes = boxes.shape[0]
    coords_per_box = boxes.shape[1]
    num_dims = coords_per_box // 2
    
    # Create IDs array (just sequential indices)
    ids = np.arange(num_boxes, dtype=np.int32)
    
    # Pre-sort the boxes by their lower bound in dimension 0
    sort_idx = np.argsort(boxes[:, 0])
    sorted_boxes = boxes[sort_idx]
    sorted_ids = ids[sort_idx]
    
    # Initialize output buffer with a reasonable initial size
    # For dense scenes, each box might collide with ~10 others on average
    estimated_collisions = min(num_boxes * 10, 10000000)  # Cap at 10 million to avoid memory issues
    pairs = np.empty((estimated_collisions, 2), dtype=np.int32)
    
    # Set initial parameters for the hybrid algorithm
    # -inf and inf for the bounding segment
    cur_lo = np.float32(-np.inf)
    cur_hi = np.float32(np.inf)
    
    # Start at the highest dimension
    cur_dim = num_dims - 1
    
    # Initialize collision count
    cur_count = 0
    
    # Make the first call to the hybrid recursive function
    # For self-collision, we call with the same set for both intervals and points
    cur_count = hybrid(
        sorted_boxes, sorted_ids,    # intervals and their IDs
        sorted_boxes, sorted_ids,    # points and their IDs (same as intervals for self-collision)
        cur_lo, cur_hi,              # initial segment spans the entire space
        cur_dim,                     # start at highest dimension
        cur_count, pairs, cutoff     # collision tracking and algorithm parameters
    )
    
    # Return only the portion of the array that contains valid collisions
    return pairs[:cur_count]
