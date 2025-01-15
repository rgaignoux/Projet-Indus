import os
from PIL import Image

def list_images_in_directory(directory_path):
    # Vérifie si le chemin spécifié est valide
    if not os.path.exists(directory_path):
        print(f"Le chemin {directory_path} n'existe pas.")
        return
    
    # Parcours des fichiers dans le répertoire
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        try:
            # Tente d'ouvrir le fichier en tant qu'image
            with Image.open(file_path) as img:
                print(file_path)
        except Exception:
            # Ignore les fichiers qui ne sont pas des images
            pass

# Exemple d'utilisation
# Remplacez 'votre/repertoire/ici' par le chemin de votre répertoire
list_images_in_directory('results\\nouveau_dataset')