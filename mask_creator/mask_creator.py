# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:36:12 2025

@author: theth
"""

import json
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageDraw
import cv2
import os

# Chargement de l'orthophotographie
def load_orthophoto(folder, file):
    print("load", folder+file)
    img  = Image.open(folder + file) 
    dims_string = file[6:-4]    # On considère que l'image a pour nom "mosaic ... .png"
    dims_string = dims_string.split("-")
    pic_min_x, pic_min_y = float(dims_string[0]), float(dims_string[1])
    pic_max_x, pic_max_y = float(dims_string[2]), float(dims_string[3])
    print("frame :", pic_min_x, pic_min_y, pic_max_x, pic_max_y)
    return img, pic_min_x, pic_min_y, pic_max_x, pic_max_y

def line_in_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
    # Cas 1 : ligne grossièrement éloignée de l'image
    if (   (line[0][0] < pic_min_x and line[1][0] < pic_min_x)
        or (line[0][0] > pic_max_x and line[1][0] > pic_max_x)
        or (line[0][1] < pic_min_y and line[1][1] < pic_min_y)
        or (line[0][1] > pic_max_y and line[1][1] > pic_max_y)):
        return False
    
    # Cas 2 : un point au moins est dans l'image
    for point in line:
        if (point[0] > pic_min_x and point[0] < pic_max_x and
            point[1] > pic_min_y and point[1] < pic_max_y):
            return True
    
    # Cas 3 : aucun point dans l'image mais la ligne traverse l'image
    # TODO
    
    return False

# Chargement des filaires
def load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
    # Charger les filaires JSON
    with open(json_file, 'r') as f:
        filaires = json.load(f)
        
    # Créer une image raster vide
    line_list = []
    

    # Ajouter les filaires à la liste
    for feature in filaires:
        meter_coords = feature.get('coordinates', [])

        lines_temp = [[[e for e in meter_coords[i]], [e for e in meter_coords[i+1]]] for i in range(len(meter_coords)-1)]
        
        for line in lines_temp:
            #print("test ligne")
            if line_in_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y):
                line_list.append(line)
    
    print(len(line_list), "lignes")
    return line_list

def projectOnImage(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, width, height):
    for i in range(len(line)):
        line[i][0] = (line[i][0]-pic_min_x) / (pic_max_x-pic_min_x)
        line[i][1] = (pic_max_y-line[i][1]) / (pic_max_y-pic_min_y)
    return line

def saveMask(polygons, im_w, im_h, mask_path):
    # Étape 1 : Créer une image noire avec Pillow
    image = Image.new("L", (im_w, im_h), 0)  # Image en niveaux de gris (0 = noir)

    # Étape 2 : Dessiner les polygones sur l'image
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        polygon[:,0] = polygon[:,0]*im_w
        polygon[:,1] = polygon[:,1]*im_h
        todraw = [(pt[0], pt[1]) for pt in polygon]
        draw.polygon(todraw, outline=255, fill=255)  # Couleur blanche (255) pour les polygones
    
    # return
    # Convertir l'image Pillow en tableau NumPy pour OpenCV
    image_np = np.array(image)
    
    # Étape 3 : Appliquer une fermeture morphologique avec OpenCV
    kernel = np.ones((5, 5), np.uint8)  # Définir un noyau carré de 5x5
    closed_image = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)
    
    # Étape 4 : Sauvegarder ou afficher les résultats
    # Convertir le résultat en image Pillow pour sauvegarde/affichage
    result_image = Image.fromarray(closed_image)
    result_image.save(mask_path)  # Sauvegarde

def drawImage(img, width, height):
    center = (width//2, height//2)
    new_height = height
    new_width = int(img.width * height / img.height)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(resized_img)
    # Affichage de l'image
    im_id = canvas.create_image(center[0], center[1], image=tk_img)
    return tk_img, im_id

# Obtenir la forme d'un segment de voie à partir des informations qui lui sont associées
def shapeFilaire(filaire, largeur, decalage):
    v_filaire = np.array([filaire[1][0] - filaire[0][0],
                 filaire[1][1] - filaire[0][1]])
    v_normal = np.array([v_filaire[1], -v_filaire[0]])/np.sqrt(np.sum(v_filaire**2))
    
    if v_normal[0] < 0: v_normal = -v_normal
    
    polygon = np.array([[ filaire[0][0] + (decalage+largeur)*v_normal[0],
                      filaire[0][1] + (decalage+largeur)*v_normal[1]    ],
                      [ filaire[1][0] + (decalage+largeur)*v_normal[0],
                      filaire[1][1] + (decalage+largeur)*v_normal[1]    ],
                      [ filaire[1][0] + (decalage-largeur)*v_normal[0],
                      filaire[1][1] + (decalage-largeur)*v_normal[1]    ],
                      [ filaire[0][0] + (decalage-largeur)*v_normal[0],
                      filaire[0][1] + (decalage-largeur)*v_normal[1]    ]])
    return polygon


def drawPoly(poly, color, w, h, highlighted=False):
    x_origin = w//2-h//2
    temp_poly = np.zeros(poly.shape)
    temp_poly[:,0] = poly[:,0]*h
    temp_poly[:,1] = poly[:,1]*h
    outline = "#f00" if highlighted else ""
    id_poly = canvas.create_polygon(temp_poly[0][0], temp_poly[0][1], temp_poly[1][0], temp_poly[1][1],
                                    temp_poly[2][0], temp_poly[2][1], temp_poly[3][0], temp_poly[3][1],
                                    fill=color, width = 2, outline=outline)
    canvas.move(id_poly, x_origin, 0)

# Boucle principale    
def loop():
    global canvas, img, tk_img, w_canvas, h_canvas, filaires, largeur, decalage, endloop
    global largeur_com, decalage_com, largeur_v, decalage_v, im_id
    # Récupére la largeur
    width = canvas.winfo_width() 
    # Récupére la hauteur
    height = canvas.winfo_height()
    
    # Vitesse de commande
    decalage_v = 0 if decalage_com == 0 else decalage_v+0.0005*decalage_com
    largeur_v = 0 if largeur_com == 0 else largeur_v+0.0005*largeur_com
    
    
    # Execution de la commande
    largeur[index] += largeur_v
    decalage[index] += decalage_v
    
    if largeur[index] < 0: largeur[index] = 0
    
    if True or (w_canvas != width or h_canvas != height):
        w_canvas, h_canvas = width, height
        canvas.delete(tk.ALL) # Nettoyage de l'écran
        tk_img, im_id = drawImage(img, width, height)
    
        current_poly = np.array([])
        for i, f in enumerate(filaires):
            poly = shapeFilaire(f, largeur[i], decalage[i])
            color = "#eee" #"#f00" if i == index else "#0f0"
            if i == index:
                current_poly = poly
            drawPoly(poly, color, width, height)
            # drawLine(l1, color, w_canvas, h_canvas)
            # drawLine(l2, color, w_canvas, h_canvas)
        if len(current_poly) != 0:
            drawPoly(current_poly, "", width, height, highlighted=True)
        
    if not endloop:
        root.after(10, loop)
    else:
        root.destroy()

# Interaction utilisateur
def on_key(event):
    global index, largeur_com, decalage_com, endloop
    print(f"Touche pressée : {event.keysym}")
    # Ajouter ici la logique pour modifier la largeur des filaires
    if event.keysym == "x" and index < largeur.shape[0]-1:
        index += 1
    if event.keysym == "w" and index > 0:
        index -= 1
    if event.keysym == "Right":
        decalage_com = 1
    if event.keysym == "Left":
        decalage_com = -1
    if event.keysym == "Up":
        largeur_com = 1
    if event.keysym == "Down":
        largeur_com = -1
    if event.keysym == "Return":
        endloop = True
    if event.keysym == "BackSpace":
        largeur_com = -1000
    print("index :", index)

def on_key_release(event):
    global largeur_com, decalage_com
    largeur_com = 0
    decalage_com = 0



folder = "images_to_segment/"
file_list = os.listdir(folder)
print(file_list)

# Fonction principale
for file in file_list:
    json_file = 'Data/filaires-projection.json'  # Chemin du fichier JSON

    # Charger les données
    img, pic_min_x, pic_min_y, pic_max_x, pic_max_y = load_orthophoto(folder, file)
    filaires = load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y)
    for e in filaires:
        e = projectOnImage(e, pic_min_x, pic_min_y, pic_max_x, pic_max_y, img.width, img.height)

    # Largeur et décalage de la voie par rapport au filaire de voie
    largeur = np.zeros(len(filaires))
    decalage = np.zeros(len(filaires))
    index = 0
    endloop = False
    largeur_com, decalage_com = 0,0
    largeur_v, decalage_v = 0,0
    # Interface Tkinter
    root = tk.Tk()
    tk_img = ImageTk.PhotoImage(img)
    im_id = 0
    
    w_canvas = 0
    h_canvas = 0
    canvas = Canvas(root, width=img.width, height=img.height)
    canvas.pack(fill = "both", expand = True)

    

    

    root.bind("<Key>", on_key)  # Pression de touche
    root.bind("<KeyRelease>", on_key_release)  # Relâchement de touche
    loop()
    
    root.mainloop()
    
    print("c'est le moment d'enregistrer les masques dans des images...")
    mask_path = "result/mask" + file[6:]

    polygons = []
    for i in range(len(filaires)):
        if largeur[i] != 0:
            polygons.append(shapeFilaire(filaires[i], largeur[i], decalage[i]))
    
    saveMask(polygons, img.width, img.height, mask_path)
