## Description
Ce projet implémente un fine-tuning du modèle DeepLabV3 pour la segmentation de routes à partir d'images satellites. Le modèle a été entraîné sur un jeu de données de 50 images et testé sur un ensemble de 22 images pour évaluer ses performances. Malgré la petite taille du jeu de données, l'objectif est de demontrer l'efficacité du fine-tuning pour des applications spécifiques et de préparer le modèle pour des améliorations futures.


## Modèle Utilisé
DeepLabV3 : Un modèle de segmentation d'image basé sur des réseaux de neurones convolutifs, capable de réaliser une segmentation au niveau des pixels avec une grande précision.

-**Backbone** : ResNet-50
-**Framework** : PyTorch

## Entraînement
Nombre d'images pour l'entraînement : 50 images
Nombre d'epochs : 10

## Résultats
**Métrique de Performance**
IoU moyen sur le jeu de test : **0.1368**
Ce score reflète les défis liés à l'entraînement sur un petit jeu de données et met en lumière les opportunités d'amélioration par l'augmentation du jeu de données.



## 🏁 Résultats obtenus après fine tuning

Voici quelques résultats obtenus  :

### 🖼️ Image 1
![Segmentation 1](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/fine_tuning/image6_segmentation_fine_tuning.png?raw=true)

### 🖼️ Image 2
![Segmentation 2](https://github.com/rgaignoux/Projet-Indus/blob/Mawa/fine_tuning/prediction_image12_fine_tuning.png?raw=true)
