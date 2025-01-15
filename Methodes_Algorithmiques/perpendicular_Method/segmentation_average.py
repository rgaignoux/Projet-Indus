import cv2
import numpy as np
import sys
import utils
import argparse
import math
import glob
import os
from draw_normals import extract_normals

def remove_outliers_and_compute_mean(widths_around, lower_percentile=2, upper_percentile=98):
    if not widths_around:
        return 0
        
    lower_bound = np.percentile(widths_around, lower_percentile)
    upper_bound = np.percentile(widths_around, upper_percentile)
    filtered_widths = []

    for w in widths_around:
        if lower_bound <= w <= upper_bound:
            filtered_widths.append(w)

    average = 0
    if filtered_widths:
        average = np.mean(filtered_widths)

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
road_paths = glob.glob(f"{directory_path}//road*.png")

for (axis_path, road_path) in zip(axes_paths, road_paths):
    print("Processing image...")

    # Load the road image
    road = cv2.imread(road_path, cv2.IMREAD_COLOR)

    # Load the central axis image
    central_axis = cv2.imread(axis_path, cv2.IMREAD_GRAYSCALE)
    central_axis = cv2.resize(central_axis, (road.shape[1], road.shape[0]))
    #central_axis = cv2.bitwise_not(central_axis) # si ancien dateset : décommenter

    # Filter out the white pixels
    hsv_image = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    hsv_image[mask > 0] = [0, 0, 160]
    road2 = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Gaussian filtering
    road_blurred = cv2.GaussianBlur(road2, (5,5), 0)
    #road_blurred = cv2.bilateralFilter(road, 9, 75, 75)

    # Canny edge detection
    edges = cv2.Canny(road_blurred, 85, 85)

    # Find the road edges using normals
    normals, points, _ = extract_normals(central_axis)
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
        k = 150
        (i, j) = pos
        start = index - k
        end = index + k

        if start < 0:
            end += abs(start)
            start = 0

        if end >= len(points):
            start -= abs(end - len(points) + 1)
            end = len(points) - 1

        # Compute the weighted average widths
        widths1_around = []
        widths2_around = []

        for idx_pt in range(start, end):
            pt = points[idx_pt]
            dist = utils.distance(pt, pos)
            if dist <= k:
                widths1_around.append(widths1[idx_pt])
                widths2_around.append(widths2[idx_pt])

        average1 = remove_outliers_and_compute_mean(widths1_around)
        average2 = remove_outliers_and_compute_mean(widths2_around)

        average_widths1[(i, j)] = average1
        average_widths2[(i, j)] = average2


    # Create the segmentation mask
    segmentation_mask = np.zeros(road.shape[:2])

    for pos in points:
        (i, j) = pos

        if math.isnan(average_widths1[pos]):
            width1 = 0
        else:
            width1 = int(math.floor(average_widths1[pos]))

        if math.isnan(average_widths2[pos]):
            width2 = 0
        else:
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
    result[central_axis >= 1] = np.array([0, 0, 0])
    result[segmentation_mask >= 1] = (0.4 * np.array([0, 255, 255]) + 0.6 * result[segmentation_mask >= 1]).astype(np.uint8)
    
    cv2.imwrite(f"perpendicular_Method//results//overlay_{filename}.png", result)
    if display == 1:
        utils.display_image("Result", result)