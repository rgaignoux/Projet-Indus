import cv2
import numpy as np
import sys
from utils import *
from skimage import io, morphology
from region_growing import *
import matplotlib.pyplot as plt
import time



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

# First Step : Mask the road image with the central axis

# Load the central axis image
central_axis = cv2.imread('images/axe0.png', cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Invert the image

# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255

# Load the road image
road = cv2.imread('images/route0.png')
road = cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
road = preprocess_image2(road, filter_size=5)  # Preprocess the image

# Apply dilation to the skeleton image
kernel = np.ones((35, 35), np.uint8)
dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

# Apply the dilated mask to the road image
masked_road = cv2.bitwise_and(road, road, mask=dilated_mask)

# Preprocess the image
img_preprocessed = preprocess_image2(masked_road, filter_size=5)

# Display image + central axis
overlay = road.copy()
overlay[skeleton == 255] = [0, 255, 0]  # Green for segmented region
display_image("Segmentation Result", overlay)

# Second Step : Apply region growing to segment the road

# Subsample the skeleton points to get the seeds
points = get_points_central_axis(skeleton)
seeds = points  # Subsample every 10th point

# Display the seeds on the road image
img_display = masked_road.copy()
for seed in seeds:
    cv2.circle(img_display, (seed[1], seed[0]), 2, (0, 255, 0), -1)

# Store X mask
stockage_mask = []
thresholds = [i for i in range(9, 30)]  # Example of different thresholds

start_time = time.time()
for idx, threshold in enumerate(thresholds):
    segmented_image = region_growing(img_preprocessed, seeds, threshold)

    # Apply morphological closing to fill the gaps
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Smooth the edges by applying a median filter
    smoothed = cv2.medianBlur(closing, 9)

    # Weight decreases as the threshold increases
    weight = 1 / (threshold + 2)

    # Store the smoothed mask with its weight
    stockage_mask.append((smoothed, weight))

# Initialize majority_mask based on the shape of the first mask
majority_mask = np.zeros_like(stockage_mask[0][0], dtype=np.uint8)

for i in range(majority_mask.shape[0]):
    for j in range(majority_mask.shape[1]):
        # Calculate the weighted sum for the current pixel
        weighted_sum = 0
        total_weight = 0

        for mask, weight in stockage_mask:
            if mask[i, j] == 255:  # If the pixel is activated (255)
                weighted_sum += weight
            total_weight += weight  # Always add the weight of the mask, even if the pixel is not activated
        
        # Majority condition with weight
        if total_weight > 0 and weighted_sum / total_weight > 0.5:
            majority_mask[i, j] = 255

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Overlay and display the final result
overlay_majority = road.copy()
overlay_majority[majority_mask == 255] = [0, 0, 255]  # Red for majority

# Display with OpenCV
cv2.imshow("Overlay Majority", overlay_majority)
cv2.imwrite("resultats/temp.png", overlay_majority)
cv2.waitKey(0)
cv2.destroyAllWindows()
