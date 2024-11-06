import cv2
import numpy as np
import matplotlib.pyplot as plt



# Fonction pour capturer le clic de souris pour choisir la graine de croissance
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed_position, seed_intensity, seeds
        seeds.append((y, x))
        cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Cliquez sur l'axe central de la route", image_display)

def region_growing(image, threshold):
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for seed in seeds:
        seed_intensity=image[seed]
        # Vérifier si la couleur de la graine a été définie
        if seed_intensity is not None:            
            # File d'attente pour les pixels à traiter dans la croissance de région
            to_process = [seed]
            
            # Croissance de région
            while to_process:
                x, y = to_process.pop(0)                
                # Vérifier si ce pixel n'a pas déjà été traité
                if mask[x, y] == 0:
                    neighbor_intensity = image[x, y]
                    # Vérifier si la couleur est dans la tolérance par rapport à la graine
                    if np.linalg.norm(np.int32(neighbor_intensity) - np.int32(seed_intensity)) < threshold:
                        # Marquer ce pixel comme appartenant à la région
                        mask[x, y] = 255
                        # Ajouter les pixels voisins à la file d'attente
                        if x > 0: to_process.append((x - 1, y))
                        if x < image.shape[0] - 1: to_process.append((x + 1, y))
                        if y > 0: to_process.append((x, y - 1))
                        if y < image.shape[1] - 1: to_process.append((x, y + 1))
            # Boucle pour vérifier chaque pixel dans le masque et remplir les petits trous
            mask_ameliore = mask.copy()
            height, width = mask.shape
            for x in range(1, height - 1):
                for y in range(1, width - 1):
                    # Vérifier si le pixel est à 0 mais entouré par au moins 2 pixels à 255
                    if mask[x, y] == 0:
                        neighbors = [
                            mask[x - 1, y],  # haut
                            mask[x + 1, y],  # bas
                            mask[x, y - 1],  # gauche
                            mask[x, y + 1]   # droite
                        ]
                        # Compter les voisins qui sont à 255
                        if neighbors.count(255) >= 2:
                            mask_ameliore[x, y] = 255  # Marquer ce pixel comme appartenant à la région
    return mask_ameliore
           
 # Charger l'image

def binariser(image,width):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_not(binary_image)
    kernel = np.ones((width, width), np.uint8)  # Noyau carré de taille 'width'
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated_image


# Orthophotographie chargement
image_path = 'image.png'
image = cv2.imread(image_path)
image_display = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB pour Matplotlib

# Variables globales pour stocker la position et la couleur de la graine
seed_intensity = None
seeds = []

# Afficher l'image pour permettre à l'utilisateur de cliquer sur un pixel central de la route
cv2.imshow("Cliquez sur l'axe central de la route", image_display)
cv2.setMouseCallback("Cliquez sur l'axe central de la route", on_mouse_click)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Obtention du mask
mask=region_growing(image,60)

# Charger l'image de l'axe central. Sert de deuxième filtre
imagePlan_path = 'plan.png'  # Remplacez par le chemin de votre image
imagePlan = cv2.imread(imagePlan_path)
plan=binariser(imagePlan,20)
plan_resized = cv2.resize(plan, (image_rgb.shape[1], image_rgb.shape[0]))


#Image à afficher
overlay = image_rgb.copy()
overlay[mask == 255] = [0, 0, 255] 
overlay_with_plan = overlay.copy()
overlay[plan_resized == 0] = image_rgb[plan_resized == 0]


# Afficher le résultat
plt.subplot(2, 3, 1)
plt.title("Image originale")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Pan")
plt.imshow(plan,cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Région segmentée (Route)")
plt.imshow(overlay)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("sans le Plan")
plt.imshow(overlay_with_plan)
plt.axis('off')
plt.show()
