import cv2
import matplotlib.pyplot as plt
import numpy as np
#Algo pour renforcer le contraste: en desosus du seuil on diminue els valeurs
# et au dessus on augmente
def generate_renforcementContraste_basique(image,seuil ):
    """
        Generation d'un renforcement de contraste d'une image
    """
    #Passage aux float pour les caluls
    image_float = image.astype(np.float32)
    if(len(image.shape)==2): #Image en niveaux de gris
        contraste=np.where(image_float < seuil, image_float * 0.2, image_float * 1.1)
    else: #Image en couleur
        contraste=np.zeros_like(image_float)
        for i in range(3):  
            contraste[:, :, i] = np.where(image_float[:, :, i] < seuil, image_float[:, :, i] * 0.5, image_float[:, :, i] * 1.1)
    contraste = np.clip(contraste, 0, 255).astype(np.uint8) #clipping et retour au unisgned
    return contraste

def affichercontraste(nom_image,seuil):
    image = cv2.imread(nom_image)
    contraste=generate_renforcementContraste_basique(image,seuil)
    if(len(image.shape)==2):
        cmap="gray"
    else:
        cmap=None
    # Afficher les images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Image Originale')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(contraste, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Image avec plus de contrastes')

    plt.show()

