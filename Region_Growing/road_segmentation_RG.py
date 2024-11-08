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

##########################################################
# First Step : Mask the road image with the central axis #
##########################################################

# Load the central axis image
central_axis = cv2.imread('images/axe_central.png', cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Invert the image

# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255

# Load the road image
road = cv2.imread('images/route.png')
road = resize_image(road)
road = preprocess_image2(road, filter_size=5) # Preprocess the image

# Apply dilation to the skeleton image
kernel = np.ones((35, 35), np.uint8)
dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

# Apply the dilated mask to the road image
masked_road = cv2.bitwise_and(road, road, mask=dilated_mask)

##########################################################
# Second Step : Apply region growing to segment the road #
##########################################################

# Subsample the skeleton points to get the seeds
points = get_points_central_axis(skeleton)
seeds = points[::5] # Subsample every 10 point

# Display the seeds on the road image
img_display = masked_road.copy()
for seed in seeds:
    cv2.circle(img_display, (seed[1], seed[0]), 2, (0, 255, 0), -1) # dumb column-major (col, row) format...
display_image("Seeds", img_display)

# Perform region growing
threshold = int(sys.argv[1])
segmented_image = region_growing(img_preprocessed, seeds, threshold)

# Overlay the segmentation result on the original image
overlay = road.copy()
overlay[segmented_image == 255] = [255, 0, 0]  # Blue color for segmented region

# Display the result
display_image("Segmentation Result", overlay)

#####################################################
# Third Step : Post-process the segmentation result #
#####################################################

# Apply morphological closing to fill the gaps
kernel = np.ones((7, 7), np.uint8)
closing = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel, iterations=3)
display_image("Closing", closing)

# Smooth the edges by applying median filter
smoothed = cv2.medianBlur(closing, 9)
display_image("Smoothed", smoothed)

# Overlay the smoothed result on the original image
overlay = road.copy()
overlay[smoothed == 255] = [255, 0, 0]  # Blue color for segmented region
display_image("Final Result", overlay)