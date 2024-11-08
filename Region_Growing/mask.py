import cv2
import numpy as np
import sys
from utils import *
from skimage import io, morphology
import matplotlib.pyplot as plt

def update_kernel_size(val):
    """
    Slider callback function to update the kernel size and apply dilation to the skeleton image.
    """
    kernel_size = val
    
    # Ensure kernel_size is odd, preferred for convolution kernels
    if kernel_size % 2 == 0: 
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

    # Apply the dilated mask to the road image
    masked_road = cv2.bitwise_and(road, road, mask=dilated_mask)
    
    # Display the result
    cv2.imshow('Masked Road', masked_road)

# Load the central axis image
central_axis = cv2.imread('images/axe_central.png', cv2.IMREAD_GRAYSCALE)

# Invert the image (if needed)
central_axis = cv2.bitwise_not(central_axis)

# Resize the image to fit the screen
central_axis = resize_image(central_axis)
display_image("Image", central_axis)

# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0

# Skeletonize the image
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255
display_image("Skeleton", skeleton)

# Load the road image
road = cv2.imread('images/route.png')

# Resize the road image to fit the screen
road = resize_image(road)

# Create a slider to adjust the kernel size (between 1 and 50)
cv2.namedWindow('Masked Road')
cv2.createTrackbar('Size', 'Masked Road', 15, 50, update_kernel_size) # Initial value is 15
update_kernel_size(15)

# Display loop
while True:
    # Wait for a key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Escape key
        break

# Close all OpenCV windows
cv2.destroyAllWindows()