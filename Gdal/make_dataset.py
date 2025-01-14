# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:09:33 2025

@author: theth
"""

import testGdal
import filaire_de_voie
import numpy.random as rand



# Définir la zone d'intérêt.
# Pour rennes : x entre 1349000 et 1360000, y entre 7220000 et 7226000
# (positions exprimées en mètres). 1 pixel <=> 0.05 m
xRennes = (1349000, 1360000)
yRennes = (7220000, 7226000)

#Résolution en mètres (m/pixel)
resolution = 0.5

# Taille d'image requise pour le modèle
desired_w = 512
desired_h = 512

dataset_size = 20


w_meter = desired_w*resolution
h_meter = desired_h*resolution
for i in range(dataset_size):
    # tire un point aléatoire dans Rennes
    aoi_min_x = rand.randint(xRennes[0], xRennes[1]-w_meter)
    aoi_min_y = rand.randint(yRennes[0], yRennes[1]-h_meter)
    aoi_max_x = aoi_min_x + w_meter
    aoi_max_y = aoi_min_y + h_meter

    # On récupère les images correspondant à la zone
    image_path, pic_min_x, pic_min_y, pic_max_x, pic_max_y = testGdal.getRelevantImages(aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y)
    print("region of interest :", pic_min_x, pic_min_y, pic_max_x, pic_max_y)
    
    print("Create the mosaic")
    # Créer la mosaïque en mémoire avec un VRT (fichier virtuel)
    vrt = testGdal.createVRT(image_path)
    
    width = int(vrt.RasterXSize * (aoi_max_x - aoi_min_x) / (pic_max_x - pic_min_x))
    height = int(vrt.RasterYSize  * (aoi_max_y - aoi_min_y) / (pic_max_y - pic_min_y))
    print(f"Picture size : {width}x{height}")


    testGdal.exportAsPng(vrt, aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y, desired_w, desired_h, folder="dataset/")
    # Nettoyage
    testGdal.endVRT()

    # Image de sortie
    
    # mask_path = ("dataset/filaires_" + str(aoi_min_x) + "-"
    #                + str(aoi_min_y) + "-" 
    #                + str(aoi_max_x) + "-"
    #                + str(aoi_max_y) + ".png")

    # filaire_de_voie.generate_road_image(mask_path,
    #                     aoi_min_x, aoi_min_y,
    #                     aoi_max_x, aoi_max_y,
    #                     desired_w, desired_h)







