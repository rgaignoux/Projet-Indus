import cv2
import numpy as np
import sys
from utils import *
from region_growing import *
from skimage import io, morphology
import matplotlib.pyplot as plt

def get_points_central_axis(img):
    """
    Get the central axis points from the skeletonized central axis image.
    """
    points = []
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i, j] > 0:
                points.append((i, j))

    return points

central_axis_path = sys.argv[1]
road_path = sys.argv[2]
threshold = int(sys.argv[3])

##########################################################
# First Step : Mask the road image with the central axis #
##########################################################

# Load the road image
road = cv2.imread(road_path)
road = scale_image(road)
road = preprocess_image2(road, filter_size = 15, sigma = 150) # Preprocess the image

# Load the central axis image
central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis, road.shape[1], road.shape[0])
central_axis = cv2.bitwise_not(central_axis)  # Invert the image

# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255

# Apply dilation to the skeleton image
kernel = np.ones((35, 35), np.uint8)
dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

# Apply the dilated mask to the road image
img_preprocessed = cv2.bitwise_and(road, road, mask=dilated_mask)

##########################################################
# Second Step : Apply region growing to segment the road #
##########################################################

# Get the seeds from all the central axis points
seeds = get_points_central_axis(skeleton)

# Display the seeds on the road image
img_display = img_preprocessed.copy()
img_display = cv2.convertScaleAbs(img_display)
for seed in seeds:
    cv2.circle(img_display, (seed[1], seed[0]), 2, (0, 255, 0), -1) # dumb column-major (col, row) format...
display_image("Seeds", img_display)

# Perform region growing
segmented_image = region_growing(img_preprocessed, seeds, threshold)

# Overlay the segmentation result on the original image
overlay = road.copy()
overlay[segmented_image == 255] = [255, 0, 0]  # Blue color for segmented region

# Display the result
display_image("Segmentation Result", overlay)

#####################################################
# Third Step : Post-process the segmentation result #
#####################################################

# Apply morphological opening to remove inexistent roads
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel, iterations=1)
display_image("Opening", opening)

# Apply morphological closing to fill the gaps
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
display_image("Closing", closing)

# Smooth the edges by applying median filter
smoothed = cv2.medianBlur(closing, 3)
display_image("Smoothed", smoothed)

# Overlay the smoothed result on the original image
overlay = road.copy()
overlay[smoothed == 255] = [255, 0, 0]  # Blue color for segmented region
display_image("Final Result", overlay)