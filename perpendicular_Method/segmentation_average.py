import cv2
import numpy as np
import sys
import utils
import argparse
import math
import glob
import os
from draw_normals import extract_normals

def remove_outliers(widths_around, average, threshold = 0.5):
    # Remove outliers using z-score method
    if np.std(widths_around) != 0:
        z_scores = np.abs((widths_around - average) / np.std(widths_around))
        widths_around = [w for w, z in zip(widths_around, z_scores) if z < threshold]

    # Update the average width
    if widths_around:
        average = np.mean(widths_around)

    return average

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str) # Directory path containing the images
parser.add_argument('-min', type=int, default=1) # Min range for edge detection
parser.add_argument('-max', type=int, default=50) # Max range for edge detection
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

    # Load the central axis image
    central_axis = cv2.imread(axis_path, cv2.IMREAD_GRAYSCALE)
    central_axis = cv2.bitwise_not(central_axis) # Invert the image

    # Bilateral filtering
    road_blurred = cv2.medianBlur(road, 9)

    # Canny edge detection
    edges = cv2.Canny(road_blurred, 50, 75)

    # Find the road edges using normals
    normals, points = extract_normals(central_axis)
    widths1 = []
    widths2 = []

    for k in range(0, len(points)):
        # Extract k-th point of the central axis
        (i, j) = points[k]

        width1 = 0
        width2 = 0

        (norm_x, norm_y) = normals[(i, j)]
        
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

        widths1.append(width1)
        widths2.append(width2)


    # Compute the average widths
    average_widths1 = {}
    average_widths2 = {}

    for index, pos in enumerate(points):
        # Extract 2*k points around the current point
        k = 75
        (i, j) = pos
        start = index - k
        end = index + k

        if start < 0:
            end += abs(start)
            start = 0

        if end >= len(points):
            start -= abs(end - len(points) + 1)
            end = len(points) - 1

        # Compute the average widths
        widths1_around = widths1[start:end]
        widths2_around = widths2[start:end]
        average1 = np.percentile(widths1_around, 50)
        average2 = np.percentile(widths2_around, 50)

        average_widths1[(i, j)] = average1
        average_widths2[(i, j)] = average2


    # Create the segmentation mask
    segmentation_mask = np.zeros(road.shape[:2])

    for pos in points:
        (i, j) = pos
        width1 = int(math.floor(average_widths1[pos]))
        width2 = int(math.floor(average_widths2[pos]))

        (norm_x, norm_y) = normals[(i, j)]

        # Flag in the normal direction
        for n in range(width1): 
            x2 = int(j + n * norm_x)
            y2 = int(i + n * norm_y)

            if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                segmentation_mask[y2, x2] = 1
            
        # Flag in the opposite direction
        for n in range(width2):
            x3 = int(j - n * norm_x)
            y3 = int(i - n * norm_y)

            if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                segmentation_mask[y3, x3] = 1


    # Post process segmentation
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
    filename = os.path.splitext(os.path.basename(road_path))[0]
    cv2.imwrite(f"perpendicular_Method//results//segm_{filename}.png", segmentation_mask)
    if display == 1:
        utils.display_image("Segmentation")

    # Overlay the segmentation on the road image
    result = road.copy()
    result[segmentation_mask >= 1] = (0.4 * np.array([0, 255, 255]) + 0.6 * result[segmentation_mask >= 1]).astype(np.uint8)
    cv2.imwrite(f"perpendicular_Method//results//overlay_{filename}.png", result)
    if display == 1:
        utils.display_image("Result", result)
