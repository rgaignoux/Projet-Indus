import cv2
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread("image.png")

# Convertir l'image de BGR à HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extraire uniquement le canal H (Hue)
hue_channel = hsv_image[:, :, 0]

# Afficher le canal H
plt.imshow(hue_channel, cmap='hsv')
plt.title("Canal H (Hue) de l'image HSV")
plt.axis('off')  # Désactiver les axes
plt.show()
