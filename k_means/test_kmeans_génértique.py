import cv2
import numpy as np
import matplotlib.pyplot as plt

#Algo par chatgpt. J'en ai un fait main, mais moins rapide.


def segmentation_Kmeans_A_with_blur(image_name, k=2, blur_ksize=(5, 5)):
    """
    Applique un flou sur l'image avant la segmentation K-means sur le canal A.
    
    :param image_name: Le nom de l'image à charger et segmenter.
    :param k: Nombre de clusters à créer (par exemple, 2 pour rouge/vert).
    :param blur_ksize: Taille du noyau du flou (k x k).
    """
    # Charger l'image
    image = cv2.imread(image_name)

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
    _, labels, centers = cv2.kmeans(A_reshaped.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruire l'image segmentée à partir des labels
    segmented_A = labels.reshape(A.shape)

    # Convertir les images de BGR à RGB pour affichage avec matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

    # Affichage du résultat de la segmentation
    plt.figure(figsize=(8, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)  # Convertir en RGB pour matplotlib
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_A, cmap='jet')  # Utilisation d'une palette de couleurs
    plt.title(f"Segmentation par K-means (k={k})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blurred_image_rgb)  # Convertir l'image floutée en RGB
    plt.title(f"Image floutée")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
image_name = 'image.png'  # Remplacer par le chemin de ton image
segmentation_Kmeans_A_with_blur(image_name, k=5, blur_ksize=(13, 13))  # 5x5 est la taille du noyau de flou
