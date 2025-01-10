import cv2
import numpy as np

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