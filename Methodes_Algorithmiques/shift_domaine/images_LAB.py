import cv2
import matplotlib.pyplot as plt

def afficher_et_enregistrer_LAB(image_name, save_path="canal_A.png"):
    """
    Affiche l'image dans l'espace colorimétrique LAB et enregistre le canal A.
    """
    # Charger l'image
    image = cv2.imread(image_name)

    # Convertir l'image de BGR à LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Séparer les canaux L, A, et B
    L, A, B = cv2.split(lab_image)
    # Normaliser les valeurs du canal A pour les rendre comprises entre 0 et 255
    A_normalized = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)
    # Afficher les canaux L, A, et B séparément
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(L, cmap='gray')  # Afficher le canal L en niveaux de gris
    plt.title("Canal L (Luminance)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(A, cmap='gray')  # Afficher le canal A en niveaux de gris
    plt.title("Canal A (Chrominance - Vert à Rouge)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(B, cmap='gray')  # Afficher le canal B en niveaux de gris
    plt.title("Canal B (Chrominance - Bleu à Jaune)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Enregistrer le canal A
    cv2.imwrite(save_path, A_normalized)  # Sauvegarder le canal A dans un fichier

# Exemple d'utilisation
image_name = 'image.png'  # Remplacer par le chemin de l'image
save_path = 'canal_A.png'  # Le chemin où tu veux enregistrer le canal A
afficher_et_enregistrer_LAB(image_name, save_path)
