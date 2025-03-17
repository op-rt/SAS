from render_utils import init_buffers, pack_color, direct_render
import numpy as np
import SAS

W, H = 1600, 900    # Dimensions of canvas
N = 10000           # Number of balls/bboxes

# Indices of the balls/bboxes
ids = np.arange(N) 

# Number of polylines and coordinates
num_coords = 8    # (p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y) for a 2d polyline forming a box/square
num_plines = N

# Initialize the buffers
coords_buffer, weights_buffer, colors_buffer = init_buffers("polyline_2d", num_plines, num_coords, is_stroked=True, is_colored=True, is_closed=True)

# Initialize associated arrays
box_verts = np.asarray(coords_buffer).reshape(num_plines, num_coords)
colors = np.asarray(colors_buffer).reshape(num_plines)
weights = np.asarray(weights_buffer).reshape(num_plines)

# Predefine stroke weights
weights[:] = np.full(num_plines, 1)

# Colors are not predefined (a box will change color when colliding)
# Just preparing their packed representations as np.uint32
red_packed = pack_color(np.array([255, 30, 40]))
blue_packed = pack_color(np.array([30, 40, 255])) 
grey_packed = pack_color(np.array([140, 140, 140])) 

# Helper array to reorder AABB coordinates into box vertices on the fly
#                = [min_x, max_y, max_x, max_y, max_x, min_y, min_x, min_y]
#                = [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]
verts_pattern_2d = np.array([0, 3, 1, 3, 1, 2, 0, 2])


def setup():
    size(W, H, P2D)
    stroke_cap(PROJECT)
    frame_rate(1000)
    text_size(16)
    fill("#000")
    
    global balls
    
    # Create balls with varying radii scattered randomly
    balls = Balls(num=N, canvas_width=W, canvas_height=H)
    
    # Change the radius of the first ball
    balls.rad[0] = 120
    
    
def draw():
    background("#FFF")
        
    # Update balls' positions
    balls.update()
    
    # First ball is set to mouse position (interactivity)
    balls.pos[0] = np.array([mouse_x, mouse_y])
            
    # Get the AABB of each ball
    AABBs = balls.get_AABB()
    
    # Find all colliding pairs (bulk load / batch query)
    pairs = SAS.query_pairs(AABBs, cutoff=5000)
    
    
    # ------------------------ RENDER ------------------------
    
    # OPTION 1. All Boxes (using Direct Buffers, too much overhead otherwise)
    
    # Group indices by event (colliding vs non-colliding)
    i1 = np.unique(pairs[:, 0])                # first collider
    i2 = np.unique(pairs[:, 1])                # second collider
    i0 = np.setdiff1d(ids, np.union1d(i1, i2)) # non-colliding
    
    # Retrieve boxes' edges from their compact representation
    # i.e. from (min_x, max_x, min_y, max_y) to (p1<->p2<->p3<->p4)    
    box_verts[:] = AABBs[:, verts_pattern_2d]
        
    # Update colors
    colors[i1] = red_packed
    colors[i2] = blue_packed 
    colors[i0] = grey_packed 
    
    # Batch render the polylines directly from the native buffers
    direct_render()
                

    # OPTION 2. Points + Lines (lighter scene with default py5 rendering pipeline)
    
#     # Boxes' centers (as points)
#     push_style()
#     stroke_weight(4)
#     points(balls.pos)
#     pop_style()
#     
#     # Colliding pairs (as lines)
#     push_style()
#     stroke(255, 30, 40)
#     lines(balls.pos[pairs].reshape(-1, 4))
#     pop_style()
    
    # -------------------------------------------------------
    
    
    # Info
    push_style()
    text("num: " + str(N), 10, 20)
    text("pairs: " + str(pairs.shape[0]), 10, 40)
    text("fps: %i" % int(get_frame_rate()), 10, 60)
    pop_style()
        

class Balls:
    
    def __init__(self, num, canvas_width, canvas_height):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        
        self.num = num
        self.pos = np.column_stack((np.random.uniform(0, canvas_width, num), np.random.uniform(0, canvas_height, num)))
        self.phi = np.random.uniform(0, 2 * np.pi, num) * .25
        self.vel = np.column_stack((np.cos(self.phi), np.sin(self.phi))) * .5
        self.rad = np.random.randint(2, 7, num)        
        
    def update(self):
        
        # Update positions 
        self.pos += self.vel
        
        # Check for out-of-bound conditions along x and y axes
        out_of_bounds = (self.pos > [self.canvas_width, self.canvas_height]) | (self.pos < [0, 0])

        # Flip the velocity where out-of-bounds conditions are met
        self.vel *= ~out_of_bounds * 2 - 1
        
    def get_AABB(self):

        # Empty array of shape (N, 4) 
        AABBs = np.empty((self.num, 4), dtype=np.int32)
        
        # Populate the array with all the bboxes
        AABBs[:, 0] = self.pos[:, 0] - self.rad
        AABBs[:, 1] = self.pos[:, 0] + self.rad
        AABBs[:, 2] = self.pos[:, 1] - self.rad
        AABBs[:, 3] = self.pos[:, 1] + self.rad

        return np.ascontiguousarray(AABBs)
