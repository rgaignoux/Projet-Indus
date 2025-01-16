Structure du code:

####################################################################
####                            Gdal                            ####
####################################################################

Le dossier Gdal contient l'ensemble des programmes nécessitant l'utilisation de la bibliothèque
Gdal.

Gdal/filaire_de_voie.py contient du code relatif à la manipulation des troncons de voie contenus
dans le fichier json à télécharger au lien suivant
https://data.rennesmetropole.fr/explore/dataset/troncons-de-voies-du-referentiel-voies-et-adresses-de-rennes-metropole/export/?location=13,48.10961,-1.6349&basemap=0a029a
et à placer dans un répertoire Gdal/Data/ .
-	latlon_to_projected(lat, lon) : projection de coordonnées d'un point du repère monde
  (latitude, longitude) dans le repère de Rennes.
-	generate_road_image() : génération d'un masque des filaires de voie. La zone d'intérêt et les
  dimensions de l'image finale sont fixés par l'utilisateur. 
-	main : Test des fonctions présentes en dessinant un masque sur une zone connue.


Gdal/projecteur_filaires.py contient le programme pour écrire une version simplifiée du fichier
contenant les filaires de voie. Les coordonnées y sont projetées dans le repère de Rennes afin
qu'il puisse être ensuite utilisé sans avoir besoin de recourir à la bibliothèque Gdal.

Gdal/testGdal.py contient des fonctions permettant la manipulation de la base de données
d'orthophotographies téléchargeable au format json à l'adresse suivante
https://data.rennesmetropole.fr/explore/dataset/orthophotographie-2021/export/?location=9,48.12016,-1.71437&basemap=0a029a
à placer dans un répertoire Gdal/Data/ .
Il nécessite en plus la création d'un répertoire Gdal/orthoPict/ pour contenir l'ensemble des
images de la base qui sont téléchargées.
-	convertToGeoTIFF(input_file, output_file, compress) : conversion d'une image du format jp2
  dans le format geotiff. Cette opération prends énormément de temps mais les opération
 	suivantes sont beaucoup plus rapides sur les fichier geotiff.
-	getRelevantImages(aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y) : obtenir l'ensemble des images
  couvrant une zone d'intérêt (dans la projection de Rennes) en les téléchargeant et en les
 	convertissant au format geotiff si nécessaire. Renvoie une liste contenant l'ensemble des
 	chemins vers les images téléchargées ainsi que le cadre délimitant les images.
-	exportAsPng() : crée une image png de dimensions désirées à partir d'une image au format vrt
  (propre à Gdal). Nécessite de spécifier les délimitation de la zone d'intérêt.
-	createVRT(image_paths) : Crée un fichier vrt à partir de la liste des chemins d'images
  nécessaires.
-	endVRT() : supprime le fichier vrt créé.
-	main : Test des fonctions en créant une image à partir d'une zone d'intérêt.

Gdal/make_dataset.py permet de générer un ensemble vues aériennes avec une précision et une
taille fixée pour l'entrainement des modèles. Les images sont extraites à des positions
aléatoires au sein d'un périmètre approximatif délimitant Rennes. Une partie commentée permet
d'obtenir en plus les masques de filaire de voie correspondants.

Gdal/test_polygones_batiments.py : Inachevé. Utilisation des géométries de bâtiments fournies
pour créer des masques et aider ainsi à la segmentation.

####################################################################
####                        mask_creator                        ####
####################################################################

Le  dossier mask_creator contient le nécessaire pour créer des masques d'images utilisés ensuite
pour l'apprentissage et l'évaluation de modèles. Le programme n'utilise pas Gdal mais nécessite
l'utilisation de opencv2 étant donné qu'une opération de fermeture est utilisée pour améliorer
le masque obtenu. Les programmes requièrent la création d'un répertoire
"mask_creator/images_to_segment/" qui contiendra uniquement les images dont on veut avoir les
masques.

mask_creator/Data/filaires-projection.json : version simplifiée de la base des filaires de voie
obtenable via le programme Gdal/projecteur_filaires.py .

mask_creator/mask_creator.py Génère une interface graphique avec quelques fonctions permettant
de dessiner des masques d'images autour des filaires de voies. Les images proposées à segmenter
sont celles placée dans le dossier "mask_creator/images_to_segment/".
-	load_orthophoto(folder, file) : lit une orthophotograpie au format png et obtient la position
  correspondante. Le nom de l'image doit être au même format que celle générées par les autres
 	programmes étant donné que la position de l'image est indiquée dans le nom.

-	line_in_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y) : définir si un filaire de
-	voie donné apparait dur l'image.
-	load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y) : Récupère dans le json
  les filaires des voies apparaissant sur l'image.
-	projectOnImage(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, width, height) : exprime les
  coordonnées des filaires relativement à l'image (coordonnées entre 0 et 1).
-	SaveMask(polygons, im_w, im_h, mask_path) : Sauvegarde le masque résultant dans le chemin
  indiqué.
-	drawImage(img, width, height) : dessine l'orthophotographie dans la fenêtre graphique en
  ajustant sa taille à la taille de fenêtre.
-	shapeFilaire(filaire, largeur, decalage) : Détermine le rectangle correspondant à un troncon
  de voie en fonction de la largeur et du décalage par rapport au filaire correspondant.
-	drawPoly(poly, color, w, h, highlighted=False) : dessine un rectangle correspondant à un
  troncon de voie dans la fenêtre graphique.
-	loop() : boucle principale de la fenêtre tkinter. gère les différentes commandes utilisateur
  et les affichages.
-	on_key(event) et on_key_release(event) : gestion des événements clavier.
-	programme principal : lecture des images présentes dans le dossier "images_to_segment/" et
  utilisation de l'éditeur pour chacune d'entre elles.

