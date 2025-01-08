import cv2
import numpy as np
import sys
import utils
import argparse
import math
import glob
import os

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str) # Directory path containing the images
parser.add_argument('-min', type=int, default=1) # Min range for edge detection
parser.add_argument('-max', type=int, default=15) # Max range for edge detection
parser.add_argument('-display', type=int, default=0) # 0 : don't display images, 1 : display images 

args = parser.parse_args()
minmax = range(args.min, args.max) # Min and max distances to check for edges
display = args.display

# Images paths
directory_path = args.dir
axes_paths = glob.glob(f"{directory_path}//axe*.png")
road_paths = glob.glob(f"{directory_path}//route*.png")

for (axis_path, road_path) in zip(axes_paths, road_paths):
    print("Processing image...")

    # Load the road image
    road = cv2.imread(road_path, cv2.IMREAD_COLOR)
    road = utils.scale_image(road)

    # Load the central axis image
    axis = cv2.imread(axis_path, cv2.IMREAD_GRAYSCALE)
    axis = utils.resize_image_v2(axis, road.shape[1], road.shape[0])
    axis = cv2.bitwise_not(axis) # Invert the image

    # Sobel filter to compute the gradient
    sobel_x = cv2.Sobel(axis, cv2.CV_64F, 1, 0, ksize=7)
    sobel_y = cv2.Sobel(axis, cv2.CV_64F, 0, 1, ksize=7)

    # Skeletonize the central axis image
    skeleton, points = utils.skeletonize_image(axis)

    # Sort the points
    points = utils.sort_points(points)

    # Bilateral filtering
    road_blurred = cv2.bilateralFilter(road, 9, 100, 100)

    # Canny edge detection
    edges = cv2.Canny(road_blurred, 75, 100)

    # Find the road edges using normals
    widths = []

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
        k = 10
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
        widths_around = widths[start:end]
        average = np.mean(widths_around)

        # Remove outliers using z-score method
        z_scores = np.abs((widths_around - average) / np.std(widths_around))
        threshold = 3
        widths_around = [w for w, z in zip(widths_around, z_scores) if z < threshold]

        # Update the average width
        if widths_around:
            average = np.mean(widths_around)

        # Store the average width
        average_widths[(i, j)] = average


    # Create the segmentation mask
    segmentation_mask = np.zeros(road.shape[:2])

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

            # Flag in the normal direction
            for n in range(width // 2): 
                x2 = int(j + n * norm_x)
                y2 = int(i + n * norm_y)

                if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                    segmentation_mask[y2, x2] = 1

            # Flag in the opposite direction
            for n in range(width // 2):
                x3 = int(j - n * norm_x)
                y3 = int(i - n * norm_y)

                if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                    segmentation_mask[y3, x3] = 1


    # Post process segmentation
    kernel = np.ones((5, 5))
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
    segmentation_mask = cv2.medianBlur(segmentation_mask, 5)
    filename = os.path.splitext(os.path.basename(road_path))[0]
    cv2.imwrite(f"perpendicular_Method//results//segm_{filename}.png", segmentation_mask)
    if display == 1:
        utils.display_image("Segmentation")

    # Overlay the segmentation on the road image
    result = road.copy()
    result[segmentation_mask == 1] = (0.4 * np.array([0, 0, 255]) + 0.6 * result[segmentation_mask == 1]).astype(np.uint8)
    cv2.imwrite(f"perpendicular_Method//results//overlay_{filename}.png", segmentation_mask)
    if display == 1:
        utils.display_image("Result", result)
