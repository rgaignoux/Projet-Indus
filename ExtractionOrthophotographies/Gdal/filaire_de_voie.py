# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:16:00 2025

@author: theth
"""

import json
import numpy as np
from osgeo import gdal, osr
from PIL import Image, ImageDraw
import time

json_file = 'Data/troncons-de-voies-du-referentiel-voies-et-adresses-de-rennes-metropole.json'  # Chemin du fichier JSON

# Fonction permettant de projeter du repère latitude longitude dans le repère
# de référence, EPSG:3948
# EPSG4326 correspond au repère WGS84 (Latitude/Longitude)
source = osr.SpatialReference()
source.ImportFromEPSG(4326)
# EPSG4326 correspond à la projection utilisée dans la BD orthographique
target = osr.SpatialReference()
target.ImportFromEPSG(3948)
# Créer un objet de transformation
transform = osr.CoordinateTransformation(source, target)
def latlon_to_projected(lat, lon):
    # Transformer les coordonnées
    x, y, _ = transform.TransformPoint(lon, lat)
    return x, y





def generate_road_image(output_path, min_x, min_y, max_x, max_y, width, height, modes=['Automobile', 'Mode doux', 'Mixte']):
    """
    Génère une image de routes à partir de filaires JSON.

    :param json_file: Chemin vers le fichier JSON contenant les filaires.
    :param output_path: Chemin de l'image de sortie.
    :param min_coords: Coordonnées minimales (lon_min, lat_min).
    :param max_coords: Coordonnées maximales (lon_max, lat_max).
    :param resolution: Résolution de l'image (largeur, hauteur).
    """
    t0 = time.time()
    # Charger les filaires JSON
    with open(json_file, 'r') as f:
        filaires = json.load(f)
        
    # Créer une image raster vide
    raster = np.zeros((height, width), dtype=np.uint8)
    
    t1 = time.time()
    print(round(t1-t0, 2), "s for loading")

    # Transformer les coordonnées géographiques en pixels
    def geo_to_pixel(lat, lon):
        x_meter, y_meter, _ = transform.TransformPoint(lon, lat)

        x = int((x_meter - min_x) / (max_x - min_x) * width)
        y = int((max_y - y_meter) / (max_y - min_y) * height)  # Inverser l'axe Y
        return x, y

    # Dessiner les filaires sur l'image
    image = Image.fromarray(raster, mode='L')
    draw = ImageDraw.Draw(image)
    t0 = time.time()
    
    hierarchie = []
    for feature in filaires:
        if feature.get("hierarchie") not in hierarchie:
            hierarchie.append(feature.get("hierarchie"))
        
        geometry = feature.get('geo_shape').get('geometry')
        if feature.get("mode") not in modes:
            continue
        if geometry.get('type') == 'MultiLineString':
            coords = geometry.get('coordinates', [])
            if len(coords)>1:
                print("Cas non traité :", geometry)
            pixel_coords = [geo_to_pixel(lon, lat) for lon, lat in coords[0]]
            
            draw.line(pixel_coords, fill=255, width=2)  # Dessiner la ligne en blanc
        else:
            print("type non traité :", geometry.get('type'))
    t1 = time.time()
    print(round(t1-t0, 2), "s for drawing")
    
    print("hierarchies :", hierarchie)

    # Sauvegarder l'image en sortie
    image.save(output_path)
    print(f"Image générée et sauvegardée : {output_path}")


# Programme principal si le fichier filaire_de_voie.py est utilisé directement
if __name__ ==  "__main__":
    # Définir la zone d'intérêt.
    # Pour rennes : x entre 1349000 et 1360000, y entre 7219000 et 7228000
    # (positions exprimées en mètres). 1 pixel <=> 0.05 m
    # Projection utilisée : EPSG:3948
    aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y = 1350000, 7220000, 1351000, 7221000
    width, height = 400, 400  # Résolution de l'image (largeur, hauteur)

    # Sélection des routes à afficher
    # Différents modes de transports proposés : ['Automobile', 'Mode doux', 'Mixte']
    modes = ['Automobile', 'Mixte']
    
    # Image de sortie
    
    output_path = ("filaires_" + str(aoi_min_x) + "-"
                   + str(aoi_min_y) + "-" 
                   + str(aoi_max_x) + "-"
                   + str(aoi_max_y) + ".png")

    generate_road_image(output_path,
                        aoi_min_x, aoi_min_y,
                        aoi_max_x, aoi_max_y,
                        width, height,
                        modes)

    print("end of program")