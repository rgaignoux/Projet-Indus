import numpy as np
import cv2
import os

# Fonction pour charger l'orthophotographie
def load_orthophoto(folder, file):
    img = cv2.imread(folder + file, cv2.IMREAD_COLOR)
    dims_string = file[6:-4]
    dims_string = dims_string.split("-")
    pic_min_x, pic_min_y = float(dims_string[0]), float(dims_string[1])
    pic_max_x, pic_max_y = float(dims_string[2]), float(dims_string[3])
    return img, pic_min_x, pic_min_y, pic_max_x, pic_max_y

# Fonction pour remplacer la végétation par la médiane des pixels voisins avec un renforcement
def replace_vegetation_with_stronger_median(image, green_lower_bound=(10, 15, 15), green_upper_bound=(100, 255, 255), window_size=7, strength=1.5):
    # Convertir l'image en HSV pour faciliter la détection des couleurs
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Créer un masque pour détecter la végétation (couleur verte)
    lower_green = np.array(green_lower_bound)  # Limites minimales pour la couleur verte
    upper_green = np.array(green_upper_bound)  # Limites maximales pour la couleur verte
    mask_vegetation = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Obtenir les dimensions de l'image
    height, width, _ = image.shape
    
    # Créer une copie de l'image pour les modifications
    modified_image = image.copy()
    
    # Parcourir tous les pixels de l'image
    for y in range(height):
        for x in range(width):
            # Si c'est un pixel de végétation
            if mask_vegetation[y, x] > 0:
                # Définir la fenêtre autour du pixel (prend une zone carrée de taille window_size x window_size)
                y_min = max(y - window_size // 2, 0)
                y_max = min(y + window_size // 2 + 1, height)
                x_min = max(x - window_size // 2, 0)
                x_max = min(x + window_size // 2 + 1, width)
                
                # Extraire la fenêtre autour du pixel
                window = image[y_min:y_max, x_min:x_max]
                
                # Calculer la médiane pour chaque canal (BGR)
                median_value = np.median(window, axis=(0, 1))
                
                # Renforcer l'effet en multipliant par un facteur de 'strength' (affecte davantage la végétation)
                modified_value = median_value * strength
                
                # Limiter les valeurs entre 0 et 255 pour éviter les débordements
                modified_value = np.clip(modified_value, 0, 255)
                
                # Remplacer le pixel de végétation par la médiane calculée et renforcée
                modified_image[y, x] = modified_value
    
    return modified_image

# Dossier contenant les images
folder = "Methodes_Algorithmiques/UI/images_to_segment/"
file_list = os.listdir(folder)

for file in file_list:
    # Charger l'image
    road, pic_min_x, pic_min_y, pic_max_x, pic_max_y = load_orthophoto(folder, file)

    # Remplacer la végétation par la médiane des pixels voisins avec renforcement
    road_with_stronger_median = replace_vegetation_with_stronger_median(road.copy(), window_size=7, strength=1.5)  # Augmenter 'strength' pour un remplacement plus fort
    

    # Filter out the white pixels
    hsv_image = cv2.cvtColor(road_with_stronger_median, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    hsv_image[mask > 0] = [0, 0, 160]
    road2 = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # Concaténer l'image originale et l'image après remplacement
    concatenated_image = cv2.hconcat([road, road2])

    # Afficher les deux images côte à côte
    cv2.imshow("Original and Stronger Vegetation Replacement", concatenated_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
