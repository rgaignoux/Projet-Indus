# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:23:37 2024

@author: theth
"""
import json
import numpy as np
from osgeo import gdal, ogr, osr


# Fonction permettant de projeter du repère latitude longitude dans le repère
# de référence, EPSG:3948
def latlon_to_projected(lat, lon):
    # EPSG4326 correspond au repère WGS84 (Latitude/Longitude)
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    # EPSG4326 correspond à la projection utilisée dans la BD orthographique
    target = osr.SpatialReference()
    target.ImportFromEPSG(3948)
    # Créer un objet de transformation
    transform = osr.CoordinateTransformation(source, target)
    # Transformer les coordonnées
    x, y, _ = transform.TransformPoint(lon, lat)
    return x, y



# Charger le fichier JSON contenant les bâtiments
with open('Data/archive_referentiel_batiment.json', 'r') as f:
    buildings_data = json.load(f)

# Définir la zone d'intérêt en coordonnées projetées
# -1.885375706101414, 48.184336843636565, -1.871104025659985, 48.175929522264504
aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y = 1350000, 7220000, 1351000, 7221000

# Structure d'un élément de buildings_data :
# {"geo_point_2d": {"lon": float, "lat": float}, 
# "geo_shape": {
#     "type": "Feature",
#     "geometry": {
#         "coordinates": [
#             [
#                 [<lon(float)>, <lat(float)>, 0.0],
#                 ...
#             ],
#             ...
#         ],
#         "type": "Polygon"
#     },
#     "properties": {}
# },
# "gml_id": "batiment_referentiel_archive.37018",
# "objectid": 37018,
# "id_rm": "35238_039817",
# "code_insee": "35238",
# "type_bati": "B\u00e2timent Priv\u00e9",
# "zmin": 40.94,
# "zmax": 49.83}

for building in buildings_data:
    
    
    

# Création d'un driver de données vectorielles en mémoire pour stocker les polygones
driver = ogr.GetDriverByName("Memory")
data_source = driver.CreateDataSource("in_memory")
layer = data_source.CreateLayer("buildings", geom_type=ogr.wkbPolygon)

# Définir le système de coordonnées (par exemple, EPSG:4326 pour WGS 84)
spatial_ref = osr.SpatialReference()
spatial_ref.ImportFromEPSG(4326)  # Remplacer par le système de projection des données si différent

# Filtrer et ajouter les géométries des bâtiments à la couche
for building in buildings_data:
    # Vérifier si le bâtiment est dans la zone d'intérêt
    coordinates = building["geo_shape"]["geometry"]["coordinates"]
    if any(aoi_min_x <= x[0] <= aoi_max_x and aoi_min_y <= x[1] <= aoi_max_y for x in coordinates[0]):
        polygon = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in coordinates[0]:
            ring.AddPoint(coord[0], coord[1])
        polygon.AddGeometry(ring)

        # Créer une feature et l'ajouter à la couche
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(polygon)
        layer.CreateFeature(feature)
        feature = None  # Libérer la mémoire de la feature

# Création d'une image raster pour la zone d'intérêt
pixel_size = 0.5  # Ajuster la taille de pixel si nécessaire
x_res = int((aoi_max_x - aoi_min_x) / pixel_size)
y_res = int((aoi_max_y - aoi_min_y) / pixel_size)

# Création de l'image raster en mémoire avec un canal pour les bâtiments
raster_driver = gdal.GetDriverByName("MEM")
raster = raster_driver.Create('', x_res, y_res, 1, gdal.GDT_Byte)
raster.SetGeoTransform((aoi_min_x, pixel_size, 0, aoi_max_y, 0, -pixel_size))

# Projection
raster_srs = osr.SpatialReference()
raster_srs.ImportFromEPSG(4326)  # Remplacer par le même EPSG que précédemment si nécessaire
raster.SetProjection(raster_srs.ExportToWkt())

# Rasterisation de la couche vectorielle sur l'image raster
gdal.RasterizeLayer(raster, [1], layer, burn_values=[255])  # Valeur de 255 pour colorer les bâtiments

# Enregistrement de l'image en PNG
output_filename = "buildings_overlay.png"
gdal.Translate(output_filename, raster, format="PNG")

print(f"Image des bâtiments enregistrée sous {output_filename}")
