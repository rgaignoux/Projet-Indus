import cv2
import numpy as np
import sys
from importlib.machinery import SourceFileLoader
import argparse
utils = SourceFileLoader('utils', './utils.py').load_module()

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-img', type=int, default=1) # Image number (1-6)
parser.add_argument('-min', type=int, default=1) # Min range for edge detection
parser.add_argument('-max', type=int, default=15) # Max range for edge detection

args = parser.parse_args()

num_image = str(args.img)
central_axis_path = f"C:\\Users\\Robin\\Documents\\Projet-Indus\\BDD-7227\\axe{num_image}.png"
road_path = f"C:\\Users\\Robin\\Documents\\Projet-Indus\\BDD-7227\\route{num_image}.png"
minmax = range(args.min, args.max) # Min and max distances to check for edges

# Load the road image
road = cv2.imread(road_path, cv2.IMREAD_COLOR)
road = utils.scale_image(road)

# Load the central axis image
central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
central_axis = utils.resize_image(central_axis, road.shape[1], road.shape[0])
central_axis = cv2.bitwise_not(central_axis) # Invert the image

# Sobel filter to compute the gradient
sobel_x = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)
sobel_y = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)

# Skeletonize the central axis image
skeleton, points = utils.skeletonize_image(central_axis)

# Bilateral filtering
road = cv2.bilateralFilter(road, 9, 100, 100)

# Canny edge detection
edges = cv2.Canny(road, 75, 100)
utils.display_image("Edges", edges)

# Overlay the skeleton on the edges image
overlay1 = utils.overlay_mask(edges, skeleton, bgr = False)
utils.display_image("Edges with Central Axis", overlay1)

# Find the road edges using normals
road_edges = np.zeros(road.shape[:2], dtype=np.uint8)

for k in range(0, len(points)):
    # Extract k-th point of the central axis
    (i, j) = points[k]
    grad_x = sobel_x[i, j]
    grad_y = sobel_y[i, j]

    if grad_x != 0 or grad_y != 0:
        # Normalize the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude

        # Check the edges in the normal direction
        for n in minmax:
            x2 = int(j + n * norm_x)
            y2 = int(i + n * norm_y)

            if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                if edges[y2, x2] >= 0:
                    road_edges[y2, x2] = 255

                if edges[y2, x2] > 0:
                    break

        # Check the edges in the opposite direction
        for n in minmax:
            x3 = int(j - n * norm_x)
            y3 = int(i - n * norm_y)

            if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                if edges[y3, x3] >= 0:
                    road_edges[y3, x3] = 255

                if edges[y3, x3] > 0:
                    break

# Overlay the road edges on the road image
overlay2 = utils.overlay_mask(overlay1, road_edges)
utils.display_image("Road Edges", overlay2)
cv2.imwrite(f"results//region_growing//route{num_image}.png", road_copy)
