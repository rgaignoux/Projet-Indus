# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:46:04 2025

@author: theth
"""

import filaire_de_voie as fdv
import json


def projeterFilaires():
    with open(fdv.json_file, 'r') as f:
        filaires = json.load(f)
    
    list_out = []
    # Extraire les coordonnées des filaires et les projeter dans le repère de Rennes
    for feature in filaires:
        geometry = feature.get('geo_shape').get('geometry')
        
        coords = geometry.get('coordinates', [])
        
        coords_list = [fdv.latlon_to_projected(lon, lat) for lon, lat in coords[0]]
        
        list_out.append({"coordinates" : coords_list})
        
        
    json_out = 'Data/filaires-projection.json'
    with open(json_out, 'w') as f:
        json.dump(list_out, f)

projeterFilaires()