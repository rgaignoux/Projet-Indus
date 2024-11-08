import cv2
import matplotlib.pyplot as plt

def afficher_YUV(image_name):
    """
    Affiche l'image dans l'espace colorimétrique YUV.
    """
    # Charger l'image
    image = cv2.imread(image_name)

    # Convertir l'image de BGR à YUV
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Séparer les canaux Y, U, et V
    Y, U, V = cv2.split(yuv_image)

    # Afficher les canaux Y, U, et V séparément
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')  # Afficher le canal Y en niveaux de gris
    plt.title("Canal Y (Luminance)")
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
afficher_YUV(image_name)
