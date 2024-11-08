import cv2
import matplotlib.pyplot as plt

def afficher_LUV(image_name):
    """
    Affiche l'image dans l'espace colorimétrique LUV.
    """
    # Charger l'image
    image = cv2.imread(image_name)

    # Convertir l'image de BGR à LUV
    luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

    # Séparer les canaux L, U, et V
    L, U, V = cv2.split(luv_image)

    # Afficher les canaux L, U, et V séparément
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(L, cmap='gray')  # Afficher le canal L en niveaux de gris
    plt.title("Canal L (Luminance)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(U, cmap='gray')  # Afficher le canal U en niveaux de gris
    plt.title("Canal U (Chrominance)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(V, cmap='gray')  # Afficher le canal V en niveaux de gris
    plt.title("Canal V (Chrominance)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
image_name = 'image.png'  # Remplacer par le chemin de l'image
afficher_LUV(image_name)
