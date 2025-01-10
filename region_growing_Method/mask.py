import cv2
import numpy as np
import sys
from utils import *
from skimage import io, morphology
import matplotlib.pyplot as plt
import argparse  # Importation du module argparse pour le parsing des arguments

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

# Parser des arguments de la ligne de commande
def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply dilation and segmentation to road images using skeleton axis.")
    parser.add_argument("-img", type=int, required=True, help="Image number to process (e.g., 1, 2, 3, ...)")
    return parser.parse_args()

# Main script
if __name__ == "__main__":
    # Parse les arguments de la ligne de commande
    args = parse_arguments()

    # Construire les chemins des images en fonction du numéro d'image
    image_number = args.img
    central_axis_path = f'images/ImagesRennes/axe{image_number}.png'
    road_image_path = f'images/ImagesRennes/route{image_number}.png'


    # Charger l'image de l'axe central
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)

    # Inverser l'image (si nécessaire)
    central_axis = cv2.bitwise_not(central_axis)

    # Redimensionner l'image pour l'adapter à l'écran
    central_axis = resize_image(central_axis)
    display_image("Image", central_axis)

    # Convertir l'image en un tableau binaire booléen pour la squelettisation
    binary_bool = central_axis > 0

    # Squelettiser l'image
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255
    display_image("Skeleton", skeleton)

    # Charger l'image de la route
    road = cv2.imread(road_image_path)

    # Redimensionner l'image de la route
    road = resize_image(road)

    # Créer un slider pour ajuster la taille du noyau (entre 1 et 50)
    cv2.namedWindow('Masked Road')
    cv2.createTrackbar('Size', 'Masked Road', 15, 50, update_kernel_size) # Valeur initiale de 15
    update_kernel_size(15)

    # Boucle d'affichage
    while True:
        # Attendre une pression de touche pour quitter
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Touche Escape
            break

    # Fermer toutes les fenêtres OpenCV
    cv2.destroyAllWindows()
