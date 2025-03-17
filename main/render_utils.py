"""
Simple module designed to help render graphical primitives (points, lines, polylines) 
in py5 by using Direct Buffers. By allocating memory in native RAM and sharing it between 
Python, Java, and OpenGL, it avoids unnecessary data copies and speeds up rendering. 
The script uses JPype to access Java classes and methods, NumPy to manipulate arrays, 
and py5 for the sketch (drawing) environment.

Takes much inspirations from the py5 documentation on Hybrid Programming by James Schmitz 
(https://py5coding.org/content/hybrid_programming.html)

# Author: sol/ub 
# Created: 2023-06-03
# Python Version: 3.11
# Context: Testing py5 rendering performance using Direct Buffers.

"""


import jpype
import numpy as np
from py5 import get_current_sketch

# Get the PApplet instance using get_current_sketch
current_sketch = get_current_sketch()

# Initialize the utils with the sketch instance
Py5Utilities = jpype.JClass("py5utils.Py5Utilities")
utils = Py5Utilities(current_sketch._instance)


def init_buffers(primitive_type, num_primitive, num_coords=None, is_stroked=False, is_colored=False, is_closed=False):

    """
    Initialize and allocate native Direct Buffers for rendering primitives.

    This function creates Direct Buffers in native memory for the given primitive type.
    It allocates a buffer for the coordinates of the primitives and optionally for stroke weights
    and colors. The buffers are then shared with the Java/OpenGL renderer via the Py5Utilities class.

    Parameters
    ----------
    primitive_type (str):            The type of primitive to render (e.g., "point_2d", "line_2d", "polyline_2d").
    num_primitive  (int):            Number of primitives to allocate buffers for.
    num_coords     (int, optional):  Number of coordinate values per primitive. Required for polyline types.
    is_stroked     (bool, optional): If True, allocate an additional buffer for stroke weights.
    is_colored     (bool, optional): If True, allocate an additional buffer for color values.
    is_closed      (bool, optional): If True, indicates that the primitive (e.g., a polyline) should be rendered as closed.

    Returns
    -------
    tuple:     A tuple (pts_buffer, weights_buffer, colors_buffer) where each element is a Direct Buffer.
               For options not enabled (stroked/colored), the corresponding buffer is None.

    """
    
    # Primitive types mapped to their coordinate counts
    coord_counts = {"point_2d": 2,  # (px, py)
                    "point_3d": 3,  # (px, py, pz)
                    "line_2d": 4,   # (p1x, p1y, p2x, p2y)
                    "line_3d": 6}   # (p1x, p1y, p1z, p2x, p2y, p2z)

    # For polylines, we need the num_coords parameter
    if primitive_type in ["polyline_2d", "polyline_3d"]:
        if num_coords is None:
            raise ValueError(f"num_coords must be provided for {primitive_type}")

    else:
        # Get the number of coordinates for standard primitive types
        num_coords = coord_counts.get(primitive_type)
        if num_coords is None:
            raise ValueError(f"Unknown primitive type: {primitive_type}")
        
    # Create Direct Buffer for Primitives
    pts_bytearray = bytearray(num_primitive * num_coords * 4) #  (4 bytes = 1 float coordinate value)
    pts_buffer = jpype.nio.convertToDirectBuffer(pts_bytearray).asFloatBuffer()
    
    # Initialize optional buffers to None
    weights_buffer = None
    colors_buffer = None

    if is_stroked:
        # Create Direct Buffer for Stroke Weights 
        weights_bytearray = bytearray(num_primitive * 4) # (4 bytes = 1 float weight value)
        weights_buffer = jpype.nio.convertToDirectBuffer(weights_bytearray).asFloatBuffer()
     
    if is_colored:
        # Create Direct Buffer for Colors
        colors_bytearray = bytearray(num_primitive * 4) # (4 bytes = 1 int color value)
        colors_buffer = jpype.nio.convertToDirectBuffer(colors_bytearray).asIntBuffer()
            
    # Allocate buffers in native memory
    utils.shareBuffers(primitive_type, pts_buffer, weights_buffer, colors_buffer, num_primitive, num_coords, is_closed)
    
    return pts_buffer, weights_buffer, colors_buffer


def direct_render():

    """
    Trigger the rendering of primitives from the shared Direct Buffers.

    This function calls the Java-side method to render all primitives stored in the shared buffers.
    It assumes that the buffers have been initialized and shared with the renderer using init_buffers.

    """

    utils.drawPrimitives()


def pack_color(c):

    """
    Pack a color array into a single 32-bit integer in ARGB format.

    The input color array should contain either 3 values (R, G, B) or 4 values (A, R, G, B).
    If only 3 values are provided, an alpha value of 255 (fully opaque) is assumed.

    Parameters
    ----------
    c (array of ints): A 1D NumPy array with 3 or 4 elements representing a color.

    Returns
    -------
    (uint32): A 32-bit unsigned integer encoding the color in ARGB format.

    """
    
    # Assuming c is [r, g, b] or [a, r, g, b]
    if c.shape[0] == 3:
        a, r, g, b = 255, c[0], c[1], c[2]  # Add alpha=255 if not provided
    else:
        a, r, g, b = c[0], c[1], c[2], c[3]
    
    # Perform operations in the unsigned 32-bit domain
    a = np.uint32(a)
    r = np.uint32(r)
    g = np.uint32(g)
    b = np.uint32(b)
    
    packed = np.uint32((a << 24) | (r << 16) | (g << 8) | b)
    
    # Reinterpret as signed 32-bit integer
    return packed.view(np.int32)