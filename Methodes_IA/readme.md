

# Modèle Deeplabv3+ 

## Description

Le code permet de charger un modèle de segmentation, de l'appliquer à des images d'un dossier spécifié et d'enregistrer les résultats sous forme d'images segmentées, de masques inversés et de cartes de probabilité. Une superposition de masque binaire coloré est appliquée sur l'image d'origine sans assombrir les zones non concernées. Les résultats sont sauvegardés dans un dossier spécifié. A noter qu'il faut obligatoirement l'utiliser sur google collab pour éviter des problémes. 

### Metriques :
Nous avons obtenue: 
- **Average Precision**: 92%
- **Average Recall**: 71%
- **Average F1 Score**: 80%
après post-processing

### Fonctionnalités :

- **Superposition de masque** : Le masque binaire est superposé sur l'image d'origine avec une transparence configurable.
- **Génération de cartes de probabilité** : Une carte de probabilité de la segmentation est générée et visualisée.
- **Traitement par lots** : Le code peut traiter toutes les images d'un dossier donné.
- **Enregistrement des résultats** : Les images segmentées, masques et cartes de probabilité sont sauvegardées dans un dossier.

## Prérequis

Avant d'exécuter le code, assurez-vous que vous disposez des bibliothèques suivantes installées :

- `torch`
- `numpy`
- `PIL` (Pillow)
- `matplotlib`
- `torchvision`

Vous pouvez installer ces dépendances via pip :

```bash
pip install torch torchvision numpy pillow matplotlib
```
Les résultats sont visibles dans le dossier deeplabv3+, dans le sous-dossier images_test. 
