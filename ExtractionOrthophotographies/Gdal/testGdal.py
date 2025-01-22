# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import json
import requests
from osgeo import gdal
import os


def convertToGeoTIFF(input_file, output_file, compress=True):
    """
    Convertit une image au format GeoTIFF en utilisant GDAL.
    
    Args:
        input_file (str): Chemin vers le fichier source.
        output_file (str): Chemin vers le fichier GeoTIFF de sortie.
        compress (bool): Applique une compression (Deflate) si True.
    """
    # Options de création pour GeoTIFF
    creation_options = ["TILED=YES"]
    if compress:
        creation_options.append("COMPRESS=DEFLATE")
    
    # Créer des options pour gdal.Translate
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=creation_options
    )
    
    # Conversion en GeoTIFF
    print(f"Conversion de {input_file} vers {output_file}...")
    gdal.Translate(output_file, input_file, options=translate_options)
    print("Conversion terminée.")


def getRelevantImages(aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y):
    # Charger le fichier JSON
    with open('Data/orthophotographie-2021.json', 'r') as f:
        data = json.load(f)
    
    # Filtrer les images dans la zone d'intérêt
    relevant_images = [
        item for item in data 
        if (item["xmin"] <= aoi_max_x and item["xmax"] >= aoi_min_x and
            item["ymin"] <= aoi_max_y and item["ymax"] >= aoi_min_y)
    ]

    pic_min_x, pic_min_y, pic_max_x, pic_max_y = aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y 
    # Télécharger les images pertinentes
    image_paths = []
    for item in relevant_images:
        
        pic_min_x = min(pic_min_x, item["xmin"])
        pic_min_y = min(pic_min_y, item["ymin"])
        pic_max_x = max(pic_max_x, item["xmax"])
        pic_max_y = max(pic_max_y, item["ymax"])
        
        url = item["url"]
        image_filename = "orthoPict/" + url.split("/")[-1]
        image_filename_geotiff = image_filename[:-4]+".tif"
        print("Getting file :", image_filename)
        try: # Vérifie si l'image a déjà été téléchargée
            f = open(image_filename, 'r')
            f.close()
            print('already existing')
        
        except:
            response = requests.get(url)
            with open(image_filename, 'wb') as img_file:
                img_file.write(response.content)
            print('done')
        
        print("Converting to geotiff")
        try: # Vérifie si l'image a déjà été téléchargée
            f = open(image_filename_geotiff, 'r')
            f.close()
            print('already existing')
        
        except:
            convertToGeoTIFF(image_filename, image_filename_geotiff)
            print('done')
        image_paths.append(image_filename_geotiff)

    return image_paths, pic_min_x, pic_min_y, pic_max_x, pic_max_y 


def exportAsPng(vrt, aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y, resized_width, resized_height, folder = ""):
    print("Create PNG file")
    # Exporter directement en PNG ou JPEG
    output_filename = (folder + "mosaic" + str(aoi_min_x) + "-" + str(aoi_min_y) + "-"
    + str(aoi_max_x) + "-" + str(aoi_max_y) + ".png")

    translate_options = gdal.TranslateOptions(
        format="PNG",
        projWin=[aoi_min_x, aoi_max_y, aoi_max_x, aoi_min_y],
        width=resized_width,
        height=resized_height,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE"]
    )
    gdal.Translate(
        output_filename,
        vrt,
        options=translate_options
    )
    os.remove(output_filename + ".aux.xml")

def createVRT(image_paths):
    # La position de chaque image est contenue dans les métadonnées du fichier jp2
    # c'est pourquoi il est inutile de les redonner à GDAL
    vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
    return gdal.BuildVRT("/vsimem/temp_mosaic.vrt", image_paths, options=vrt_options)

def endVRT():
    gdal.Unlink("/vsimem/temp_mosaic.vrt")



# Programme principal si le fichier testGdal.py est utilisé directement
if __name__ ==  "__main__":
    # Définir la zone d'intérêt.
    # Pour rennes : x entre 1349000 et 1360000, y entre 7219000 et 7228000
    # (positions exprimées en mètres). 1 pixel <=> 0.05 m
    # Projection utilisée : EPSG:3948
    aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y = 1350000, 7220000, 1351000, 7221000

    image_paths, pic_min_x, pic_min_y, pic_max_x, pic_max_y = getRelevantImages(aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y)
    print("Create the mosaic")
    # Créer la mosaïque en mémoire avec un VRT (fichier virtuel)
    vrt = createVRT(image_paths)
    
    width = int(vrt.RasterXSize * (aoi_max_x - aoi_min_x) / (pic_max_x - pic_min_x))
    height = int(vrt.RasterYSize  * (aoi_max_y - aoi_min_y) / (pic_max_y - pic_min_y))
    print(f"Picture size : {width}x{height}")
    
    resize_factor = float(input('Choose resize factor [0 ; 1] : '))
    if not 0 < resize_factor <= 1:
        print("ERROR : Wrong resize factor")
    resized_width = int(width * resize_factor)
    resized_height = int(height * resize_factor)
    print(f"New picture size : {resized_width}x{resized_height}")
    
    if 0 < resize_factor <= 1 and input("Proceed ? (y, n) ") == "y":
        exportAsPng(vrt, aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y, resized_width, resized_height)
    print("Clear")
    # Nettoyage
    endVRT()
    
    
    print('End of program')