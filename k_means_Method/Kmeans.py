import cv2
import numpy as np
from skimage import io, morphology
import matplotlib.pyplot as plt
import argparse


# Command-line argument parsing
parser = argparse.ArgumentParser(description="Segmentation and analysis of road images using K-means and skeletonization.")
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

def get_image_paths(img_number):
    """
    Dynamically generate image paths based on the provided image number.
    """
    central_axis_path = f'images/ImagesRennes/axe{img_number}.png'
    road_path = f'images/ImagesRennes/route{img_number}.png'
    return road_path, central_axis_path

def apply_iterative_opening(mask, iterations=5, kernel_size=(5, 5)):
    """
    Apply iterative morphological opening to remove small protrusions in the mask.
    :param mask: Binary mask for opening operation.
    :param iterations: Number of iterations for opening.
    :param kernel_size: Size of the kernel for morphological operation.
    :return: Mask after iterative opening.
    """
    kernel = np.ones(kernel_size, np.uint8)

    # Apply iterative opening to remove noise or small structures
    for _ in range(iterations):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def get_points_central_axis(img):
    """
    Retrieve points from the central axis (skeleton of the axis image).
    :param img: Skeletonized central axis image.
    :return: List of points in the central axis.
    """
    points = []
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i, j] > 0:
                points.append((i, j))
    return points

def segmentation_Kmeans_A_with_blur(image, k=2, blur_ksize=(5, 5)):
    """
    Apply Gaussian blur and then perform K-means clustering on the A channel of the LAB color space.
    :param image: Input image to segment.
    :param k: Number of clusters (e.g., 2 for road and background).
    :param blur_ksize: Size of the blur kernel.
    :return: Segmented image based on K-means clustering.
    """
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, blur_ksize, 0)

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    L, A, B = cv2.split(lab_image)
    
    # Reshape the A channel for K-means clustering
    A_reshaped = A.reshape((-1, 1))

    # Apply K-means clustering to the A channel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(A_reshaped.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruct the segmented image from the labels
    segmented_A = labels.reshape(A.shape)

    return segmented_A

def get_dominant_color_in_roadKmeans(kmeans_labels, skeleton_mask):
    """
    Identify the dominant cluster representing the road using the central axis.
    :param kmeans_labels: Labels obtained from K-means clustering.
    :param skeleton_mask: Binary skeleton mask of the central axis.
    :return: The dominant cluster label corresponding to the road.
    """
    # Get pixels corresponding to the road using the skeleton mask
    road_pixels = kmeans_labels[skeleton_mask > 0]
    
    # Identify the most frequent cluster (dominant cluster)
    dominant_cluster = np.bincount(road_pixels).argmax()
    return dominant_cluster

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
    # Get paths for the road and central axis images
    road_path, central_axis_path = get_image_paths(args.img)

    # Load and preprocess the central axis image
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
    central_axis = resize_image(central_axis)  # Resize to match road image dimensions
    central_axis = cv2.bitwise_not(central_axis)  # Invert the image

    # Convert to binary (boolean) for skeletonization
    binary_bool = central_axis > 0
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255

    # Dilate the skeleton to improve mask robustness
    kernel = np.ones((35, 35), np.uint8)
    dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

    # Load the road image and preprocess it
    road = cv2.imread(road_path)
    road = cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
    road = preprocess_image2(road, filter_size=5)  # Apply preprocessing (e.g., denoising)

    ##########################################################
    # K-means Segmentation on the A channel
    ##########################################################

    # Apply K-means segmentation on the A channel of LAB color space
    mask_kmeans = segmentation_Kmeans_A_with_blur(road, k=5, blur_ksize=(13, 13))  # 5 clusters, larger blur kernel

    # Identify the dominant cluster corresponding to the road
    label_road = get_dominant_color_in_roadKmeans(mask_kmeans, skeleton)

    # Create a binary mask for the road based on the dominant cluster
    mask_kmeans_binary = np.where(mask_kmeans == label_road, 255, 0).astype(np.uint8)

    # Apply the dilated mask to the binary K-means mask
    mask_kmeans_binary = cv2.bitwise_and(mask_kmeans_binary, mask_kmeans_binary, mask=dilated_mask)

    # Apply iterative opening to remove small protrusions
    mask_kmeans_binary_opened = apply_iterative_opening(mask_kmeans_binary, iterations=40, kernel_size=(9, 9))

    # Display the processed mask after opening
    display_image("K-means Mask After Opening", mask_kmeans_binary_opened)

    # Visualization: overlay the binary mask on the road image
    overlay2 = road.copy()
    overlay2[mask_kmeans_binary == 255] = [255, 0, 0]  # Highlight segmented region in red
    overlay3 = road.copy()
    overlay3[mask_kmeans_binary_opened == 255] = [255, 0, 0]  # Highlight opened region in red

    # Display both original and processed overlays
    display_images_in_one_window("Original vs Processed Overlay", [overlay2, overlay3])
