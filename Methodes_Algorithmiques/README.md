# Projet-Indus
L'objectif du projet industriel proposé par l'entreprise Someware est de créer un modèle capable d'estimer les largeurs des rues et trottoirs afin d’accompagner les municipalités dans leurs choix d'aménagements


# Voici une liste de fonctions à appeler et leurs effets:
## Algorithmes globaux
### Segmentation de l'image via kmeans (k=5)
```bash
python .\k_means_Method\Kmeans.py -img <image_number>
```
### Segmentation de l'image via Fuzzy_c_means (k=5)
```bash
python3 .\k_means_Method\Fuzzy_c_means.py -img <image_number>
```

## Algorithmes basé Region Growing:
### affichage de l'extraction de la route
```bash
python3 .\region_growing_Method\mask.py -img 0
```
### Application d'un regin growing, avec selection des graines à la main
```bash
python .\region_growing_Method\Region_Growing.py <threshold_value> <filter_size>
```
### Application d'un region growing, avec selection des graines automatique et random 
```bash
python3 .\region_growing_Method\road_segmentation_RG.py -img 0 -thresh 15
```
### Utilisation de probabilités pour créer une segmentation
```bash
python3 .\region_growing_Method\road_segmentation_Proba.py
```

## Algorithmes basé contour
### Affichage d'une segmentation basée sur les normales à l'axe central
Utilisation de la distance à l'axe central et d'une largeur moyenne de la route pour tenter d'améliorer le résultat
```bash
 python3 ./perpendicular_Method/segmentation_normals.py -img 0 -min 75 -max 125
 ```

### Affichage d'une segmentation basée sur les normales à l'axe central + calcul de la largeur moyenne de la route en des points du contour de la route 
```bash
python3 .\perpendicular_Method\segmentation_average.py -dir 'images/ImagesRennes' -min 75 -max 125 -display 1
```
