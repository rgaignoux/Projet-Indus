# segmentation_average.py

```
python3 .\perpendicular_Method\segmentation_average.py -dir="your_images_directory"
```
```
python3 .\perpendicular_Method\segmentation_average.py -dir="images/ImagesRennes"
```
- Preprocessing de la route (filtre gaussien)
- Calcul du gradient de l'axe central afin obtenir les normales en chaque point
![alt text](results/normales.png)
- Création du masque de segmentation :
    - on applique un filtre de Canny sur l'image de route pour obtenir les contours
    - pour chaque point de l'axe central, on parcous les pixels dans la direction de la normale et on marque les pixels comme faisant partie du masque jusqu'à ce qu'on atteigne un pixel de contour
    - on moyenne selon les largeurs voisines

## Résultat :

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
