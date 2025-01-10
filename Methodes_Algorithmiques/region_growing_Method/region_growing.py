import cv2
import numpy as np
import sys
from utils import *

def preprocess_image1(img):
    """
    Preprocess the image by converting to LAB and returning the A channel.
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channel_a = img_lab[:, :, 1]
    
    # Normalize
    channel_a = cv2.normalize(channel_a, None, 0, 255, cv2.NORM_MINMAX)

    return channel_a

def preprocess_image2(img, filter_size = 5):
    """
    Preprocess the image by applying bilateral filtering.
    """
    img_preprocessed = cv2.bilateralFilter(img, filter_size, 50, 50)

    return img_preprocessed

# Global variable
seeds = []  # List to store seed positions

def on_mouse_click(event, x, y, flags, param):
    """
    Mouse callback function to capture seed points when the user left-clicks on the image.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append((y, x))  # Append the coordinates (row, col) format
        # Draw a circle on the image where the seed is added
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Select Seeds", img_display)

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <threshold_value> <filter_size>")
        sys.exit(1)

    # Parsing arguments from command line
    threshold = int(sys.argv[1])
    filter_size = int(sys.argv[2])

    # Load the road image
    img = cv2.imread('images/ImagesRennes/route0.png')

    # Resize the image to fit the screen
    img = resize_image(img)
    img_display = img.copy()

    # Display the image to select seed points
    img_display = img.copy()
    cv2.namedWindow("Select Seeds")
    cv2.setMouseCallback("Select Seeds", on_mouse_click)
    display_image("Select Seeds", img_display)

    # Preprocess the image
    img_preprocessed = preprocess_image2(img, filter_size)
    display_image("Preprocessed Image", img_preprocessed)

    # Perform region growing
    segmentation_result = region_growing(img_preprocessed, seeds, threshold)

    # Overlay the segmentation result on the original image
    overlay = img.copy()
    overlay[segmentation_result == 255] = [255, 0, 0]  # Blue color for segmented region

    # Display the final result
    display_image("Segmentation Result", overlay)
