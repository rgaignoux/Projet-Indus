# segmentation_average.py

```
python3 .\perpendicular_Method\segmentation_average.py -dir="your_images_directory"
```
```
python3 .\perpendicular_Method\segmentation_average.py -dir="images/ImagesRennes"
```

## Détails de la méthode

1. **Preprocessing :**
- Suppression des pixels proches du blanc (pouvant correspondre a des passages piétons, pistes cyclables, marquages au sol, etc.) afin d'éviter de faciliter la detection des contours de route sur l'**image de route**
- Application d'un filtre gaussien (5 x 5) sur l'**image de route** pour éliminer le bruit et fondre les pixels entre eux

2. **Calcul des normales à partir de l'axe central :**
- **Skeletonization** de l'image de l'axe central et calcul de la normal en chaque point via filtres de Sobel 5x5
- **Postprocess** des normales (correction des directions + remplacement des outliers)

![alt text](results/normales.png)

3. **Calcul des longueurs en chaque point de l'axe central skeletonizé :**
- Calcul des contours de l'image de route preprocess, via filtre de Canny
- Les pixels sont parcourus dans la direction des **2 normales** (dans la direction de la normale + dans la direction opposée à la normale) et marqués comme étant de la route jusqu-à ce qu'un contour soit rencontré

4. **Moyennage par fenetre glissante :**
- Une fois les longueurs en chaque point calculées, un moyennage est effectuée sur les 200 points autour du point
- Les outliers sont supprimés dans le calcul de la moyenne via suppression des valeurs < 2ème percentile et > 98ème percentile

5. **Post process de la segmentation :**
- Fermeture 3x3

## Métriques

- **Average Precision :** `0.8290`
- **Average Recall :** `0.8559`
- **Average F1 Score :** `0.8410`

## Segmentations :

![alt text](results/overlay_route0.png)
![alt text](results/segm_route0.png)

![alt text](results/overlay_route1.png)
![alt text](results/segm_route1.png)

![alt text](results/overlay_route2.png)
![alt text](results/segm_route2.png)

![alt text](results/overlay_route3.png)
![alt text](results/segm_route3.png)

![alt text](results/overlay_route4.png)
![alt text](results/segm_route4.png)

![alt text](results/overlay_route5.png)
![alt text](results/segm_route5.png)

![alt text](results/overlay_route6.png)
![alt text](results/segm_route6.png)

![alt text](results/overlay_route7.png)
![alt text](results/segm_route7.png)

![alt text](results/overlay_route8.png)
![alt text](results/segm_route8.png)

![alt text](results/overlay_route9.png)
![alt text](results/segm_route9.png)

![alt text](results/overlay_route10.png)
![alt text](results/segm_route10.png)

![alt text](results/overlay_route11.png)
![alt text](results/segm_route11.png)

![alt text](results/overlay_route12.png)
![alt text](results/segm_route12.png)

![alt text](results/overlay_route13.png)
![alt text](results/segm_route13.png)

![alt text](results/overlay_route14.png)
![alt text](results/segm_route14.png)

![alt text](results/overlay_route15.png)
![alt text](results/segm_route15.png)

![alt text](results/overlay_route16.png)
![alt text](results/segm_route16.png)

![alt text](results/overlay_route17.png)
![alt text](results/segm_route17.png)

![alt text](results/overlay_route18.png)
![alt text](results/segm_route18.png)

![alt text](results/overlay_route19.png)
![alt text](results/segm_route19.png)

![alt text](results/overlay_route20.png)
![alt text](results/segm_route20.png)

![alt text](results/overlay_route21.png)
![alt text](results/segm_route21.png)