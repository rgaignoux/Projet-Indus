import cv2
import numpy as np
import sys
import argparse
from skimage import morphology
from importlib.machinery import SourceFileLoader
utils = SourceFileLoader('utils', './utils.py').load_module()

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-img', type=int, default=1) # Image number (1-6)
args = parser.parse_args()

num_image = str(args.img)
central_axis_path = f"C:\\Users\\Robin\\Documents\\Projet-Indus\\BDD-7227\\axe{num_image}.png"

# Load the central axis image
central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
central_axis = utils.scale_image(central_axis)
central_axis = cv2.bitwise_not(central_axis) # Invert the image

# Skeletonize the central axis image
skeleton, points = utils.skeletonize_image(central_axis)
skeleton_copy = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

# Sobel filter to compute the gradient
sobel_x = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)
sobel_y = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)

# Draw the normals
step = 5 # Draw normals every 5 points
line_length = 15

for k in range(0, len(points), step):
    # Extract k-th point of the central axis
    (i, j) = points[k]
    grad_x = sobel_x[i, j]
    grad_y = sobel_y[i, j]

    if grad_x != 0 or grad_y != 0:
        # Normalize the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude

        # Compute the line coordinates
        x1 = j
        y1 = i
        x2 = int(j + line_length * norm_x)
        y2 = int(i + line_length * norm_y)

        # Compute the line coordinates in the opposite direction
        x3 = int(j - line_length * norm_x)
        y3 = int(i - line_length * norm_y)

        # Draw the normal lines
        cv2.circle(skeleton_copy, (x1, y1), 3, (255, 0, 0), -1)
        cv2.line(skeleton_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(skeleton_copy, (x1, y1), (x3, y3), (0, 255, 0), 2)
        

utils.display_image("Normals", skeleton_copy)
