# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:36:12 2025

@author: theth
"""

import json
import numpy as np
import cv2
import os
from PIL import Image

# Chargement de l'orthophotographie
def load_orthophoto(folder, file):
    print("load", folder+file)
    img = cv2.imread(folder + file)
    dims_string = file[6:-4]  # On considère que l'image a pour nom "mosaic ... .png"
    dims_string = dims_string.split("-")
    pic_min_x, pic_min_y = float(dims_string[0]), float(dims_string[1])
    pic_max_x, pic_max_y = float(dims_string[2]), float(dims_string[3])
    print("frame :", pic_min_x, pic_min_y, pic_max_x, pic_max_y)
    return img, pic_min_x, pic_min_y, pic_max_x, pic_max_y

def line_in_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
    if ((line[0][0] < pic_min_x and line[1][0] < pic_min_x) or
        (line[0][0] > pic_max_x and line[1][0] > pic_max_x) or
        (line[0][1] < pic_min_y and line[1][1] < pic_min_y) or
        (line[0][1] > pic_max_y and line[1][1] > pic_max_y)):
        return False

    for point in line:
        if (pic_min_x < point[0] < pic_max_x and pic_min_y < point[1] < pic_max_y):
            return True

    # Cas 3 : aucun point dans l'image mais la ligne traverse l'image
    # TODO: Vous pouvez ajouter ici la logique pour gérer ce cas.

    return False

# Chargement des filaires
def load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
    with open(json_file, 'r') as f:
        filaires = json.load(f)

    routes_list = []

    for feature in filaires:
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

import random
def generate_random_color():
    # Fixer des valeurs proches de (0, 255, 0)
    red = random.randint(100, 255)  # Rouge faible
    green = random.randint(100, 255)  # Vert dominant
    blue = random.randint(100, 255)  # Bleu faible
    return (red, green, blue)


folder = "images_to_segment/"
file_list = os.listdir(folder)

# Fonction principale
for file in file_list:
    json_file = 'Data/filaires-projection.json'  # Chemin du fichier JSON

    # Charger les données
    img, pic_min_x, pic_min_y, pic_max_x, pic_max_y = load_orthophoto(folder, file)
    routes = load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y)
    print(routes)
    # Dessiner les lignes
    for route in routes:
        #une couleur pour la routes
        random_color = generate_random_color()
        #on affiche toutes les lignes de la route
        for line in route:
            projected_line = project_on_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, img.shape[1], img.shape[0])
            for i in range(len(projected_line) - 1):
                cv2.line(img, projected_line[i], projected_line[i + 1], random_color, 2)
        cv2.imshow('Segments', img)
        cv2.waitKey(0)
         
    # Afficher l'image avec les segments
    cv2.imshow('Segments', img)
    cv2.waitKey(0)  # Attendre une touche pour continuer
    cv2.destroyAllWindows()
