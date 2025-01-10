import cv2
import numpy as np
import sys
import argparse
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

def get_image_paths(img_number):
    """
    Dynamically generate image paths based on the provided image number.
    """
    road_path = f'images/ImagesRennes/route{img_number}.png'
    central_axis_path = f'images/ImagesRennes/axe{img_number}.png'
    return road_path, central_axis_path

# Fonction pour analyser les arguments en ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Process and segment road images.")
    parser.add_argument('-img', type=int, default=0, choices=[0, 1], help="Select image: 0 for axe0, 1 for axe1")
    parser.add_argument('-thresh', type=int, default=15, help="Threshold for region growing segmentation")
    return parser.parse_args()

# Charger et prétraiter l'image
def load_and_preprocess_images(axe_num):
    # Get the dynamic paths based on the selected image number
    road_path, central_axis_path = get_image_paths(axe_num)
    
    # Load the central axis image
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
    central_axis = resize_image(central_axis)
    central_axis = cv2.bitwise_not(central_axis)  # Invert the image

    # Convert to binary and skeletonize
    binary_bool = central_axis > 0
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255

    # Load and preprocess road image
    road = cv2.imread(road_path)
    road = cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
    road = preprocess_image2(road, filter_size=5)
    
    return central_axis, road, skeleton

# Principal script
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load and preprocess the images based on the selected option
    central_axis, road, skeleton = load_and_preprocess_images(args.img)
    
    # Dilation of the skeleton
    kernel = np.ones((35, 35), np.uint8)
    dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

    # Apply the dilated mask to the road image
    masked_road = cv2.bitwise_and(road, road, mask=dilated_mask)

    # Preprocess the image for region growing
    img_preprocessed = preprocess_image2(masked_road, filter_size=5)

    # Get points on the central axis and set the seeds for region growing
    points = get_points_central_axis(skeleton)
    seeds = points[::5]  # Subsample every 5th point

    # Perform region growing segmentation
    segmented_image = region_growing(img_preprocessed, seeds, args.thresh)

    # Overlay segmentation result on original road image
    overlay = road.copy()
    overlay[segmented_image == 255] = [255, 0, 0]  # Color the segmented area in red

    # Show the segmentation result
    display_image("Segmentation Result", overlay)

    # Post-processing of the segmentation result (closing and smoothing)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    smoothed = cv2.medianBlur(closing, 5)

    # Final result overlay
    overlay = road.copy()
    overlay[smoothed == 255] = [255, 0, 0]  # Red color for the smoothed region
    display_image("Final Result", overlay)
