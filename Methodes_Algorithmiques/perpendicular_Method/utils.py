import cv2
import numpy as np
from skimage import morphology
import math
def display_image(window_name, img):
    """
    Display an image in a window with the given name.
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

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
def resize_image_v2(img, width, height):
    """
    Resize the image to the given width and height.
    """
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    
    return img

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

def scale_image(img, scale_percent = 65):
    """
    Scale the image by the given scale percentage.
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return img


def skeletonize_image(img):
    """
    Skeletonize the given image.
    """
    # Convert the image to a binary boolean array to skeletonize it
    binary_bool = img > 0

    # Skeletonize the image
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255

    # Exract the points from the skeleton
    points = get_points_central_axis(skeleton)

    return skeleton, points


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


def sort_points(points):
    """
    Sort the points of the central axis based on their proximity to each other.
    """
    # Start with the first point in the list
    sorted_points = [points.pop(0)]

    while points:
        last_point = sorted_points[-1]

        # Find the nearest point to the last added point
        nearest_point = min(points, key=lambda p: distance(last_point, p))
        sorted_points.append(nearest_point)
        points.remove(nearest_point)

    return sorted_points

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def overlay_mask(image, mask, bgr = True):
    """
    Overlay a mask on an image.
    """
    if not bgr:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    return overlay