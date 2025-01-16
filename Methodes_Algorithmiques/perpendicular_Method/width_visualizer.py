import cv2
import numpy as np
import sys
import utils
import argparse
import math
import glob
import os
from draw_normals import extract_normals

def draw_circle_and_normal(i, j, norm_x, norm_y, img, width1, width2):
    if norm_x != 0 or norm_y != 0:
        x1 = j
        y1 = i
        x2 = int(j + width1 * norm_x)
        y2 = int(i + width1 * norm_y)
        x3 = int(j - width2 * norm_x)
        y3 = int(i - width2 * norm_y)

        cv2.circle(img, (x1, y1), 2, (255, 0, 0), -1)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(img, (x1, y1), (x3, y3), (0, 255, 0), 2)

    return img

def get_width_meters(i, j, norm_x, norm_y, segm, resolution = 5):
    # Check the edges in the normal direction
    width1 = 0
    for n in range(200):
        x2 = int(j + n * norm_x)
        y2 = int(i + n * norm_y)

        if x2 >= 0 and x2 < segm.shape[1] and y2 >= 0 and y2 < segm.shape[0]:
            width1 += 1
            if segm[y2, x2] < 255:
                break

    # Check the edges in the opposite direction
    width2 = 0
    for n in range(200):
        x3 = int(j - n * norm_x)
        y3 = int(i - n * norm_y)

        if x3 >= 0 and x3 < segm.shape[1] and y3 >= 0 and y3 < segm.shape[0]:
            width2 += 1
            if segm[y3, x3] < 255:
                break

    width_px = width1 + width2
    width_m = width_px / resolution # 5px = 1m

    return width1, width2, width_m

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str) # Directory path containing the images
args = parser.parse_args()

# Images paths
directory_path = args.dir
road_paths = ["C:\\Users\\Robin\\Documents\\ESIR3\\Projet-Indus\\Methodes_Algorithmiques\\images\\ImagesRennes\\road_1353000_7222800_1353200_7223000.png"]
axes_paths = ["C:\\Users\\Robin\\Documents\\ESIR3\\Projet-Indus\\Methodes_Algorithmiques\\images\\ImagesRennes\\axe_1353000_7222800_1353200_7223000.png"]
segm_paths = ["C:\\Users\\Robin\\Documents\\ESIR3\\Projet-Indus\\Methodes_Algorithmiques\\perpendicular_Method\\results\\nouveau_dataset\\segm_road_1353000_7222800_1353200_7223000.png"]

for (axis_path, segm_path, road_path) in zip(axes_paths, segm_paths, road_paths):
    # Read road, axis and segm image
    road = cv2.imread(road_path, cv2.IMREAD_COLOR)
    axis = cv2.imread(axis_path, cv2.IMREAD_GRAYSCALE)
    segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

    old_dataset = False
    if old_dataset:
        axis = cv2.resize(axis, (road.shape[1], road.shape[0]))
        axis = cv2.bitwise_not(axis)

    # Extract normals
    normals, points, _ = extract_normals(axis)

    # Get width at index 0
    current_idx = 0
    cv2.namedWindow("Road Image") 
    
    while True:
        # Get width in meters at current_idx
        (i, j) = points[current_idx]
        (norm_x, norm_y) = normals[(i, j)]
        width1, width2, width_m = get_width_meters(i, j, norm_x, norm_y, segm)
        print(f"Width in meters at point {i}, {j} is : {width_m} meters")

        # Draw circles and normals
        road_draw = draw_circle_and_normal(i, j, norm_x, norm_y, road.copy(), width1, width2)
        road_draw = utils.resize_image(road_draw, scale_percent=60)
        cv2.imshow("Road Image", road_draw)
        
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # Press 'n' to go to the next image
            current_idx += 1
        elif key == ord('q'):  # Press 'q' to quit
            break

cv2.destroyAllWindows()