import cv2
import numpy as np
import sys, os
import argparse
from skimage import io, morphology
import matplotlib.pyplot as plt
import skfuzzy as fuzz  # Importer la bibliothèque fuzzy

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Segmentation and analysis of road images using Fuzzy C-Means and skeletonization.")
parser.add_argument('-img', type=int, required=True, help="Image number (e.g., 0, 1, etc.)")
args = parser.parse_args()


def resize_image(img, scale_percent = 65):
    """
    Resize the image by the given scale percentage.
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return img

def preprocess_image2(image, filter_size=5):
    """
    Preprocess the image by applying Gaussian blur.
    
    :param image: Input image to preprocess.
    :param filter_size: Size of the kernel for Gaussian blur.
    :return: Blurred image.
    """
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    
    return blurred_image

# Define paths to the images based on the input image number
def get_image_paths(img_number):
    """
    Dynamically generate image paths based on the provided image number.
    """
    road_path = f'images/ImagesRennes/route{img_number}.png'
    central_axis_path = f'images/ImagesRennes/axe{img_number}.png'
    return road_path, central_axis_path

def apply_iterative_opening(mask, iterations=5, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    for _ in range(iterations):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_points_central_axis(img):
    points = []
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i, j] > 0:
                points.append((i, j))
    return points

def segmentation_fcm_A_with_blur(image, c=2, blur_ksize=(5, 5)):
    """
    Apply a blur to the image and perform Fuzzy C-Means segmentation on the A channel of the LAB model.
    
    :param image: Image to segment.
    :param c: Number of clusters (e.g., 2 for red/green segmentation).
    :param blur_ksize: Size of the blur kernel (k x k).
    """
    # Apply blur to the image before segmentation
    blurred_image = cv2.GaussianBlur(image, blur_ksize, 0)

    # Convert the blurred image from BGR to LAB
    lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab_image)  # Use only the A channel
    
    # Reshape the A channel to a single column for FCM
    A_reshaped = A.reshape((-1, 1)).T  # Shape must be (1, n_pixels)
    
    # Apply the Fuzzy C-Means algorithm
    cntr, u, _, _, _, _, _ = fuzz.cmeans(A_reshaped.astype(np.float64), c, 2, error=0.005, maxiter=1000)
    
    # u is the membership matrix of shape (c, n_pixels)
    # We can use the maximum membership to determine the assigned cluster
    u_max = np.argmax(u, axis=0).reshape(A.shape)

    return u_max

def get_dominant_color_in_roadFCM(fcm_labels, skeleton_mask):
    """
    Identify the dominant cluster associated with the road using the central axis with Fuzzy C-Means.
    """
    road_pixels = fcm_labels[skeleton_mask > 0]
    dominant_cluster = np.bincount(road_pixels).argmax()  # Most frequent cluster in the central axis
    return dominant_cluster


def display_image(window_name, img):
    """
    Display an image in a window with the given name.
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def display_images_in_one_window(window_name, images):
    """
    Display multiple images in a single window.

    :param window_name: The name of the window where images will be displayed.
    :param images: A list of images to display.
    """
    # Vérifie si toutes les images ont la même hauteur
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    
    # Ajuste toutes les images à la hauteur maximale
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]  # Calcul du rapport d'aspect
        new_width = int(max_height * aspect_ratio)  # Redimensionner pour garder le rapport d'aspect
        resized_img = cv2.resize(img, (new_width, max_height))  # Redimensionner
        resized_images.append(resized_img)
    
    # Concaténer les images horizontalement
    combined_image = np.concatenate(resized_images, axis=1)

    # Afficher l'image combinée
    cv2.imshow(window_name, combined_image)
    cv2.waitKey(0)  # Attendre qu'une touche soit pressée
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    # Get paths based on the input image number
    road_path, central_axis_path = get_image_paths(args.img)

    # Load the central axis image
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
    central_axis = resize_image(central_axis)
    central_axis = cv2.bitwise_not(central_axis)  # Invert the image

    # Convert the image to a binary matrix and skeletonize it
    binary_bool = central_axis > 0
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255

    # Apply dilation to the skeleton image
    kernel = np.ones((35, 35), np.uint8)
    dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

    # Load the road image
    road = cv2.imread(road_path)
    road = cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
    road = preprocess_image2(road, filter_size=5)  # Preprocess the image

    # Generate the Mask with Fuzzy C-Means
    mask_fcm = segmentation_fcm_A_with_blur(road, c=5, blur_ksize=(13, 13))  # 5 clusters for segmentation
    
    # Get the dominant color of the road
    label_road = get_dominant_color_in_roadFCM(mask_fcm, skeleton)
    
    # Filter the Fuzzy C-Means result
    mask_fcm_binary = np.where(mask_fcm == label_road, 255, 0).astype(np.uint8)
    mask_fcm_binary = cv2.bitwise_and(mask_fcm_binary, mask_fcm_binary, mask=dilated_mask)

    # Apply openings to remove overflow
    mask_fcm_binary_opened = apply_iterative_opening(mask_fcm_binary, iterations=40, kernel_size=(9, 9))
    display_image("mask fcm reduced", mask_fcm_binary_opened)

    # Test display
    overlay2 = road.copy()
    overlay2[mask_fcm_binary == 255] = [255, 0, 0]  # Blue for the segmented region
    overlay3 = road.copy()
    overlay3[mask_fcm_binary_opened == 255] = [255, 0, 0]  # Blue for the segmented region after opening
    display_images_in_one_window("without opening and with", [overlay2, overlay3])
