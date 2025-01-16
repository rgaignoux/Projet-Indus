# Projet-Indus
Objectif du Projet
Ce projet, proposé par l'entreprise Someware, vise à développer un modèle d'intelligence artificielle capable de segmenter les routes et les trottoirs à partir d'images satellitaires. L'objectif est d'estimer les largeurs des rues et trottoirs pour aider les municipalités dans leurs décisions d'aménagement.

## Méthodologie

Deux approches principales ont été utilisées pour la segmentation des images :

-**U-Net**

-**DeepLabV3+**

J'ai aussi exploiré une piste de segmentation avec **ArcGIS Pro** en utilisant des **librairies Deep Learning**.

Ces modèles sont entraînés pour segmenter les routes et générer des masques indiquant leurs emplacements.


## 🏁 Résultats de la segmentation avec ArcGIS

Voici quelques résultats obtenus après segmentation des routes avec ArcGIS :

### 🖼️ Image 1
![Segmentation 1](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/Resultats_ArcGIS_segmentation/route11_segmentee.png?raw=true)

### 🖼️ Image 2
![Segmentation 2](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/Resultats_ArcGIS_segmentation/route15_segmented.png?raw=true)

### 🖼️ Image 3
![Segmentation 3](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/Resultats_ArcGIS_segmentation/route16_segmented.png?raw=true)


### 🖼️ Image 4
![Segmentation 4](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/Resultats_ArcGIS_segmentation/route17_segmented.png?raw=true)

### 🖼️ Image 5
![Segmentation 5](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/Resultats_ArcGIS_segmentation/route4_segmented.png?raw=true)

### 🖼️ Image 6
![Segmentation 6](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/Resultats_ArcGIS_segmentation/route3_segmented.png?raw=true)



## Description

 Ces resultats proviennent d'un modèle de segmentation développé avec ArcGIS pour extraire et analyser les routes à partir d'images satellites. Nous chargeons un modèle de segmentation créé avec ArcGIS,  l'appliquons à des images d'un dossier spécifié, et d'enregistrons les résultats sous différentes formes : images segmentées. Une superposition de masque binaire coloré est appliquée sur l'image d'origine sans assombrir les zones non concernées. Les résultats sont sauvegardés dans un dossier spécifié.

### Metriques :
Nous avons obtenue: 
- **Average Precision**: 0.88
- **Average Recall**: 0.8325
- **Average F1 Score**: 0.84

### Fonctionnalités :
- **Génération de masques** : Création de masques binaires à partir des images satellites en utilisant un modèle de segmentation pré-entraîné avec ArcGIS.
- **Superposition de masque** : Le masque binaire est superposé sur l'image d'origine avec une transparence configurable pour une meilleure visualisation..


## Prérequis

Avant d'exécuter le code, assurez-vous que vous disposez des bibliothèques suivantes installées   :

- `torch`
- `torchvision`
- `arcgis (Python API d'ArcGIS)`

## Utilisation
-**Préparation du Modèle** : Le modèle de segmentation doit être entraîné et exporté à l'aide d'ArcGIS.

## Estimations typiques du temps d'inférence

Sur GPU  :

Pour des images de taille 512x512 : environ 50 à 100 ms par image.
Pour des images de taille 1024x1024 : environ 100 à 200 ms par image.

Sur CPU :

Pour des images de taille 512x512 : environ 500 ms à 1 seconde par image.
Pour des images de taille 1024x1024 : environ 1 à 2 secondes par image.


