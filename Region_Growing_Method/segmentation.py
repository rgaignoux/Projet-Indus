import cv2
import numpy as np
import sys
from importlib.machinery import SourceFileLoader
import argparse
utils = SourceFileLoader('utils', './utils.py').load_module()

def region_growing(image, seeds, threshold):
    """
    Perform region growing from given seed points based on intensity similarity.
    """
    # Create an output image to visualize the segmented region
    segmented_image = np.zeros(image.shape[:2], dtype=np.uint8)

    for seed in seeds:
        # Initialize the region growing queue with the seed point
        queue = [seed]
        seed_intensity = image[seed]
        segmented_image[seed] = 255  # Mark the current pixel as part of the region
        while len(queue) > 0:
            current_pixel = queue.pop(0) # (row, col) format
            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:

                    if dy == 0 and dx == 0:
                        continue # Skip the current pixel

                    # Get the neighbor pixel coordinates
                    ny, nx = current_pixel[0] + dy, current_pixel[1] + dx # (row, col) format

                    # Ensure we are within image bounds
                    if ny >= 0 and ny < image.shape[0] and nx >= 0 and nx < image.shape[1]:

                        # If the pixel is not already part of the segmented region
                        if segmented_image[ny, nx] == 0:
                            neighbor_intensity = image[ny, nx]

                            # Check if the intensity difference is within the threshold
                            if np.linalg.norm(np.int32(neighbor_intensity) - np.int32(seed_intensity)) < threshold:
                                
                                queue.append((ny, nx)) # Add the neighbor to the queue for further exploration of his neighbors
                                segmented_image[ny, nx] = 255  # Mark as processed and part of the region

    return segmented_image

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-img', type=int, default=1) # Image number (1-6)
parser.add_argument('-thresh', type=int, default=20) # Threshold for region growing
parser.add_argument('-ksize1', type=int, default=5) # Closing kernel size
parser.add_argument('-ksize2', type=int, default=15) # Median filter kernel size

args = parser.parse_args()

num_image = str(args.img)
central_axis_path = f"C:\\Users\\Robin\\Documents\\Projet-Indus\\BDD-7227\\axe{num_image}.png"
road_path = f"C:\\Users\\Robin\\Documents\\Projet-Indus\\BDD-7227\\route{num_image}.png"
threshold = args.thresh
ksize1 = args.ksize1
ksize2 = args.ksize2

##########################################################
# First Step : Mask the road image with the central axis #
##########################################################

# Load the road image
road = cv2.imread(road_path)
road = utils.scale_image(road)
road_preprocess = utils.preprocess_image2(road, filter_size = 15, sigma = 150) # Preprocess the image

# Load the central axis image
central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
central_axis = utils.resize_image(central_axis, road.shape[1], road.shape[0])
central_axis = cv2.bitwise_not(central_axis)  # Invert the image

# Skeletonize the central axis image
skeleton, points = utils.skeletonize_image(central_axis)

# Apply dilation to the skeleton image
kernel = np.ones((35, 35), np.uint8)
dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

# Apply the dilated mask to the road image
img_preprocessed = cv2.bitwise_and(road_preprocess, road_preprocess, mask=dilated_mask)

##########################################################
# Second Step : Apply region growing to segment the road #
##########################################################

# Get the seeds from all the central axis points
seeds = points

# Display the seeds on the road image
img_display = road.copy()
img_display = cv2.convertScaleAbs(img_display)
for seed in seeds:
    cv2.circle(img_display, (seed[1], seed[0]), 1, (0, 255, 0), -1) # dumb column-major (col, row) format...
utils.display_image("Seeds", img_display)

# Perform region growing
segmentation = region_growing(img_preprocessed, seeds, threshold)

# Overlay the segmentation result on the original image
overlay = road.copy()
overlay[segmentation == 255] = [255, 0, 0]  # Blue color for segmented region

# Display the result
utils.display_image("Segmentation Result", overlay)

#####################################################
# Third Step : Post-process the segmentation result #
#####################################################

# Apply closing to the segmentation result
kernel = np.ones((ksize1, ksize1), np.uint8)
closing = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel)

# Smooth the edges by applying median filter
smoothed = cv2.medianBlur(closing, ksize2)
utils.display_image("Smoothed", smoothed)

# Overlay the smoothed result on the original image
overlay = utils.overlay_mask(road, smoothed, bgr = True)
utils.display_image("Final Result", overlay)