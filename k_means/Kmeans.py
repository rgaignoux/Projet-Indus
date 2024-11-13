import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','Region_Growing'))
from region_growing import *
from skimage import io, morphology
import matplotlib.pyplot as plt
from utils import *

path_route='../Region_Growing/images/route0.png'
path_axe='../Region_Growing/images/axe0.png'

def apply_iterative_opening(mask, iterations=5, kernel_size=(5, 5)):
    """
    Applique une ouverture itérative sur le masque pour supprimer les excroissances.
    :param mask: Le masque binaire sur lequel l'ouverture sera appliquée.
    :param iterations: Le nombre d'itérations de l'opération d'ouverture.
    :param kernel_size: La taille du noyau utilisé pour l'opération morphologique.
    :return: Le masque après application de l'ouverture itérative.
    """
    # Créer un noyau pour l'opération morphologique
    kernel = np.ones(kernel_size, np.uint8)

    # Appliquer l'ouverture itérative
    for _ in range(iterations):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

# Fonction pour récupérer les points de l'axe central
def get_points_central_axis(img):
    """
    Récupère les points de l'axe central (squelette de l'image de l'axe).
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
    Applique un flou sur l'image avant la segmentation K-means sur le canal A. Domaine Lab
    
    :param image: L'image à segmenter.
    :param k: Nombre de clusters à créer (par exemple, 2 pour rouge/vert).
    :param blur_ksize: Taille du noyau du flou (k x k).
    """
    # Appliquer un flou à l'image avant la segmentation
    blurred_image = cv2.GaussianBlur(image, blur_ksize, 0)

    # Convertir l'image floutée de BGR à LAB
    lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)

    # Séparer les canaux L, A, et B
    L, A, B = cv2.split(lab_image)
    

    # Reshaper le canal A en une seule colonne
    A_reshaped = A.reshape((-1, 1))

    # Appliquer K-means sur le canal A
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(A_reshaped.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruire l'image segmentée à partir des labels
    segmented_A = labels.reshape(A.shape)

    return segmented_A

def get_dominant_color_in_roadKmeans(kmeans_labels,skeleton_mask):
    """
    Identifie le cluster dominant associé à la route en utilisant l'axe central.
    """
    road_pixels = kmeans_labels[skeleton_mask > 0]
    dominant_cluster = np.bincount(road_pixels).argmax()  # Cluster le plus fréquent dans l'axe central
    return dominant_cluster


                   
# Load the central axis image
central_axis = cv2.imread(path_axe, cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Invert the image

# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255
# Apply dilation to the skeleton image
kernel = np.ones((35, 35), np.uint8)
dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

# Load the road image
road = cv2.imread(path_route)
road=cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
road = preprocess_image2(road, filter_size=5) # Preprocess the image


                    ##########################################################
                        # Generation du Mask par K_means
                    ##########################################################

#Application du kmeans au canal A du domaine LAB
mask_kmeans=segmentation_Kmeans_A_with_blur(road, k=5, blur_ksize=(13, 13))  # 5x5 est la taille du noyau de flou
#Obtention de la couleur représentant la route 
label_road=get_dominant_color_in_roadKmeans(mask_kmeans,skeleton)
#Filtrage du k-means
mask_kmeans_binary = np.where(mask_kmeans == label_road, 255, 0).astype(np.uint8)
mask_kmeans_binary = cv2.bitwise_and(mask_kmeans_binary, mask_kmeans_binary, mask=dilated_mask)
#Ouvertures pour tenter de supprimer le débordement
mask_kmeans_binary_opened = apply_iterative_opening(mask_kmeans_binary, iterations=40, kernel_size=(9, 9))
display_image("mask kmeans réduit",mask_kmeans_binary_opened)

#Test d'affichage 
overlay2 = road.copy()
overlay2[mask_kmeans_binary == 255] = [255, 0, 0]  # Blue color for segmented region
overlay3 = road.copy()
overlay3[mask_kmeans_binary_opened == 255] = [255, 0, 0]  # Blue color for segmented region
display_images_in_one_window("sans ouverture et avec",[overlay2,overlay3])