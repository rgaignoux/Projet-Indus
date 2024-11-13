import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','Region_Growing'))
from region_growing import *
from skimage import io, morphology
import matplotlib.pyplot as plt
import skfuzzy as fuzz  # Importer la bibliothèque fuzzy
from utils import *

path_route='../Region_Growing/images/route0.png'
path_axe='../Region_Growing/images/axe0.png'

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
    Applique un flou sur l'image avant la segmentation Fuzzy C-Means (FCM) sur le canal A du modèle LAB
    
    :param image: L'image à segmenter.
    :param c: Nombre de clusters (ex. 2 pour une segmentation rouge/vert).
    :param blur_ksize: Taille du noyau de flou (k x k).
    """
    # Appliquer un flou à l'image avant la segmentation
    blurred_image = cv2.GaussianBlur(image, blur_ksize, 0)

    # Convertir l'image floutée de BGR à LAB
    lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab_image)  # On prend uniquement le canal A
    
    # Reshaper le canal A en une seule colonne pour l'algorithme FCM
    A_reshaped = A.reshape((-1, 1)).T  # La forme doit être (1, n_pixels)
    
    # Appliquer l'algorithme Fuzzy C-Means
    cntr, u, _, _, _, _, _ = fuzz.cmeans(A_reshaped.astype(np.float64), c, 2, error=0.005, maxiter=1000)
    display_image("mask fcm réduit", u[0].reshape(A.shape))
    display_image("mask fcm réduit", u[1].reshape(A.shape))
    display_image("mask fcm réduit", u[2].reshape(A.shape))
    display_image("mask fcm réduit", u[3].reshape(A.shape))
    display_image("mask fcm réduit", u[4].reshape(A.shape))
    # u est la matrice d'appartenance, de forme (c, n_pixels)
    # On peut utiliser l'appartenance maximale pour déterminer le cluster assigné
    u_max = np.argmax(u, axis=0).reshape(A.shape)

    return u_max

def get_dominant_color_in_roadFCM(fcm_labels, skeleton_mask):
    """
    Identifie le cluster dominant associé à la route en utilisant l'axe central avec Fuzzy C-Means.
    """
    road_pixels = fcm_labels[skeleton_mask > 0]
    dominant_cluster = np.bincount(road_pixels).argmax()  # Cluster le plus fréquent dans l'axe central
    return dominant_cluster


# Charger l'image de l'axe central
central_axis = cv2.imread(path_axe, cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Inverser l'image

# Convertir l'image en une matrice binaire et la squelettiser
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255

# Appliquer une dilation à l'image du squelette
kernel = np.ones((35, 35), np.uint8)
dilated_mask = cv2.dilate(skeleton, kernel, iterations=1)

# Charger l'image de la route
road = cv2.imread(path_route)
road = cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
road = preprocess_image2(road, filter_size=5)  # Prétraiter l'image


# Générer le Mask avec Fuzzy C-Means
mask_fcm = segmentation_fcm_A_with_blur(road, c=5, blur_ksize=(13, 13))  # 5 clusters pour la segmentation
# Obtenir la couleur dominante de la route
label_road = get_dominant_color_in_roadFCM(mask_fcm, skeleton)
# Filtrage du Fuzzy C-Means
mask_fcm_binary = np.where(mask_fcm == label_road, 255, 0).astype(np.uint8)
mask_fcm_binary = cv2.bitwise_and(mask_fcm_binary, mask_fcm_binary, mask=dilated_mask)

# Appliquer des ouvertures pour supprimer les débordements
mask_fcm_binary_opened = apply_iterative_opening(mask_fcm_binary, iterations=40, kernel_size=(9, 9))
display_image("mask fcm réduit", mask_fcm_binary_opened)

# Test d'affichage
overlay2 = road.copy()
overlay2[mask_fcm_binary == 255] = [255, 0, 0]  # Bleu pour la région segmentée
overlay3 = road.copy()
overlay3[mask_fcm_binary_opened == 255] = [255, 0, 0]  # Bleu pour la région segmentée après ouverture
display_images_in_one_window("sans ouverture et avec", [overlay2, overlay3])
