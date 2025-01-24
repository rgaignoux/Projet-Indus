import cv2
import numpy as np
from skimage import morphology
import math
import os

def display_image(window_name, img, scale = False):
    """
    Display an image in a window with the given name.
    """
    if scale:
        img = resize_image(img)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def resize_image(img, scale_percent = 65):
    """
    Resize the image by the given scale percentage.
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return img

def resize_image_v2(img, width, height):
    """
    Resize the image to the given width and height.
    """
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    
    return img

def display_images_in_one_window(window_name, images):
    """
    Display multiple images in a single window.

    :param window_name: The name of the window where images will be displayed.
    :param images: A list of images to display.
    """
    # Vérifie si toutes les images ont la même hauteur
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    
    # Ajuste toutes les images à la hauteur maximale
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]  # Calcul du rapport d'aspect
        new_width = int(max_height * aspect_ratio)  # Redimensionner pour garder le rapport d'aspect
        resized_img = cv2.resize(img, (new_width, max_height))  # Redimensionner
        resized_images.append(resized_img)
    
    # Concaténer les images horizontalement
    combined_image = np.concatenate(resized_images, axis=1)

    # Afficher l'image combinée
    cv2.imshow(window_name, combined_image)
    cv2.waitKey(0)  # Attendre qu'une touche soit pressée
    cv2.destroyAllWindows()

def scale_image(img, scale_percent = 65):
    """
    Scale the image by the given scale percentage.
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return img


def skeletonize_image(img, invert=False):
    """
    Skeletonize the given image.
    """
    if invert:
        img = cv2.bitwise_not(img.astype(np.uint8))
        
    # Convert the image to a binary boolean array to skeletonize it
    binary_bool = img > 0

    # Skeletonize the image
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255

    # Exract the points from the skeleton
    points = get_points_central_axis(skeleton)

    return skeleton, points


def get_points_central_axis(img):
    """
    Get the central axis points from the skeletonized central axis image.
    """
    points = []
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i, j] > 0:
                points.append((i, j))

    return points


def sort_points(points):
    """
    Sort the points of the central axis based on their proximity to each other.
    """
    # Start with the first point in the list
    sorted_points = [points.pop(0)]

    while points:
        last_point = sorted_points[-1]

        # Find the nearest point to the last added point
        nearest_point = min(points, key=lambda p: distance(last_point, p))
        sorted_points.append(nearest_point)
        points.remove(nearest_point)

    return sorted_points

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def overlay_mask(image, mask, bgr = True):
    """
    Overlay a mask on an image.
    """
    if not bgr:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    return overlay

def remove_outliers_and_compute_mean(widths_around, threshold = 1.75):
    if not widths_around or widths_around.count(1) >= 0.8 * len(widths_around):
        return 0

    widths_around = [w for w in widths_around if w != 1]
    # Remove outliers using z-score method
    average = np.mean(widths_around)
    if np.std(widths_around) != 0:
        z_scores = np.abs((widths_around - average) / np.std(widths_around))
        widths_around = [w for w, z in zip(widths_around, z_scores) if z < threshold]

    # Update the average width
    if widths_around:
        average = np.mean(widths_around)

    return int(average)

########################## Fonctions pour l'orthophotographie ##########################

def get_min_max(file):
    base_name = os.path.splitext(os.path.basename(file))[0]  # Récupère le nom de fichier sans extension
    dims_string = base_name.split("_")[1:]  # On suppose que le nom de fichier contient "minX_minY_maxX_maxY"
    if len(dims_string) != 4:
        raise ValueError("Le nom du fichier ne contient pas les dimensions attendues au format 'minX_minY_maxX_maxY'.")
    pic_min_x, pic_min_y, pic_max_x, pic_max_y = map(float, dims_string)
    return pic_min_x, pic_min_y, pic_max_x, pic_max_y

def line_in_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
    # Cas 1 : La ligne est complètement hors de l'image
    if ((line[0][0] < pic_min_x and line[1][0] < pic_min_x) or
        (line[0][0] > pic_max_x and line[1][0] > pic_max_x) or
        (line[0][1] < pic_min_y and line[1][1] < pic_min_y) or
        (line[0][1] > pic_max_y and line[1][1] > pic_max_y)):
        # Les deux points sont entièrement hors des limites de l'image
        return False

    # Cas 2 : Au moins un point de la ligne est dans l'image
    if ((pic_min_x <= line[0][0] <= pic_max_x and pic_min_y <= line[0][1] <= pic_max_y) or
        (pic_min_x <= line[1][0] <= pic_max_x and pic_min_y <= line[1][1] <= pic_max_y)):
        # Au moins un des deux points est dans les limites de l'image
        return True

    # Cas 3 : La ligne traverse l'image sans qu'aucun point ne soit dedans
    borders = [
        [(pic_min_x, pic_min_y), (pic_min_x, pic_max_y)],  # Bord gauche
        [(pic_min_x, pic_max_y), (pic_max_x, pic_max_y)],  # Bord supérieur
        [(pic_max_x, pic_max_y), (pic_max_x, pic_min_y)],  # Bord droit
        [(pic_max_x, pic_min_y), (pic_min_x, pic_min_y)],  # Bord inférieur
    ]
    def segments_intersect(seg1, seg2):
        def ccw(A, B, C):
            # Calcule l'orientation relative des points A, B, C
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = seg1
        C, D = seg2
        # Deux segments s'intersectent si leurs extrémités sont dans des orientations opposées
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    # Vérifie si la ligne traverse un des bords de l'image
    for border in borders:
        if segments_intersect(line, border):
            return True

    # Si aucun des cas n'est vérifié, la ligne est hors de l'image
    return False

#retourne une liste de routes qui sont dans l'image
import json
def load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
    with open(json_file, 'r') as f:
        filaires = json.load(f)

    routes_list = []

    for feature in filaires:
        if (
            feature.get("etat") == "Définitif" and
            feature.get("mode") == "Automobile"
        ):
            route_list = []
            meter_coords = feature.get('coordinates', [])
            lines_temp = [[[e for e in meter_coords[i]], [e for e in meter_coords[i+1]]] for i in range(len(meter_coords)-1)]

            for line in lines_temp:
                if line_in_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
                    route_list.append(line)
            if(len(route_list) > 0):
                routes_list.append(route_list)
    
    return routes_list

def project_on_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, width, height):
    projected_line = []
    for point in line:
        x = int((point[0] - pic_min_x) / (pic_max_x - pic_min_x) * width)
        y = int((pic_max_y - point[1]) / (pic_max_y - pic_min_y) * height)
        projected_line.append((x, y))
    return projected_line