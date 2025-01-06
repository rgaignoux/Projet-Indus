import cv2
import numpy as np
import sys
import math
from importlib.machinery import SourceFileLoader
import argparse
utils = SourceFileLoader('utils', './utils.py').load_module()

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-img', type=int, default=1) # Image number (1-6)
parser.add_argument('-min', type=int, default=1) # Min range for edge detection
parser.add_argument('-max', type=int, default=20) # Max range for edge detection
parser.add_argument('-voisins', type=int, default=25) # Number of neighbors to consider in averaging

args = parser.parse_args()

num_image = str(args.img)
voisins = args.voisins
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

# Sort the points
points = utils.sort_points(points)

# Bilateral filtering
road_preprocess = cv2.bilateralFilter(road, 9, 60, 60)

# Canny edge detection
edges = cv2.Canny(road_preprocess, 75, 125)
utils.display_image("Edges", edges)

# Overlay the skeleton on the edges image
overlay1 = utils.overlay_mask(edges, skeleton, bgr = False)
utils.display_image("Edges with Central Axis", overlay1)

# Find the road edges using normals
widths  = []

for k in range(0, len(points)):
    # Extract k-th point of the central axis
    (i, j) = points[k]
    grad_x = sobel_x[i, j]
    grad_y = sobel_y[i, j]

    width1 = 0
    width2 = 0

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
                width1 += 1
                if edges[y2, x2] > 0:
                    break

        # Check the edges in the opposite direction
        for n in minmax:
            x3 = int(j - n * norm_x)
            y3 = int(i - n * norm_y)

            if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                width2 += 1
                if edges[y3, x3] > 0:
                    break

    widths.append(width1 + width2)


# Compute the average widths
average_widths = {}

for index, pos in enumerate(points):
    # Extract 2*k points around the current point
    k = voisins
    (i, j) = pos
    start = index - k
    end = index + k

    if start < 0:
        end += abs(start)
        start = 0

    if end >= len(points):
        start -= abs(end - len(points) + 1)
        end = len(points) - 1

    # Compute the average width
    widths_around = np.array(widths[start:end])
    average = np.mean(widths_around)

    # Remove outliers using z-score method
    threshold = 0.75

    if np.std(widths_around) != 0:
        z_scores = np.abs((widths_around - average) / np.std(widths_around))
        mask = z_scores < threshold
        widths_around = widths_around[mask]

        # Update the average width
        if len(widths_around) > 0:
            average = np.mean(widths_around)

    # print(f"Average width at point {pos} is {average}")
    average_widths[(i, j)] = average
    

# Display the average widths (with lines)
road_copy = road.copy()
for pos in points:
    (i, j) = pos
    width = int(math.floor(average_widths[pos]))

    grad_x = sobel_x[i, j]
    grad_y = sobel_y[i, j]

    if grad_x != 0 or grad_y != 0:
        # Normalize the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude

        # Drawn in the normal direction
        for n in range(width // 2): 
            x2 = int(j + n * norm_x)
            y2 = int(i + n * norm_y)

            if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                road_copy[y2, x2] = (0, 165, 255)

                # Check 8 neighbors
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        x_n = x2 + x
                        y_n = y2 + y

                        if x_n >= 0 and x_n < edges.shape[1] and y_n >= 0 and y_n < edges.shape[0]:
                            road_copy[y_n, x_n] = (0, 165, 255)

        # Draw in the opposite direction
        for n in range(width // 2):
            x3 = int(j - n * norm_x)
            y3 = int(i - n * norm_y)

            if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                road_copy[y3, x3] = (0, 165, 255)

                # Check 8 neighbors
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        x_n = x3 + x
                        y_n = y3 + y

                        if x_n >= 0 and x_n < edges.shape[1] and y_n >= 0 and y_n < edges.shape[0]:
                            road_copy[y_n, x_n] = (0, 165, 255)


utils.display_image("Lines", road_copy)
cv2.imwrite(f"results//perpendicular//route{num_image}.png", road_copy)

    

    
    



