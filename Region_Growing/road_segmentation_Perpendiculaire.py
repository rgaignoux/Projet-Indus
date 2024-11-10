import cv2
import numpy as np
import sys
from utils import *
from region_growing import *
from skimage import io, morphology

def get_points_central_axis(img):
    """
    Get the central axis points from the skeletonized central axis image.
    """
    points = []
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i, j] > 0:
                points.append((i, j))

    return points

def get_normal_direction(Gx, Gy, x, y):
    """
    Returns the normal direction as (d1, d2), where d1 and d2 are either 1 or -1.
    The direction is derived from the gradient components (Gx, Gy).
    The normal is perpendicular to the gradient.
    """
    d1 = -Gy[x, y]  # Perpendicular to Gy (invert the vertical component)
    d2 = Gx[x, y]   # The horizontal component stays the same

    return (d1/max(abs(d1),abs(d2)), d2/max(abs(d1),abs(d2)))


# Load the central axis image
central_axis = cv2.imread('images/axe2.png', cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Invert the image
# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255
cv2.imshow("Road", skeleton)
cv2.waitKey(0)
# Load the road image
road = cv2.imread('images/route2.png')
road=cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
road = preprocess_image2(road, filter_size=5) # Preprocess the image
# Convertir l'image "road" en niveaux de gris
road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
# Appliquer un flou gaussien pour réduire le bruit
blurred_road = cv2.GaussianBlur(road_gray, (3, 3), 0)
# Appliquer le filtre Canny pour obtenir les contours
edges_map =cv2.Canny(blurred_road, threshold1=130, threshold2=200)
cv2.imshow("Normal Vectors", edges_map)
cv2.waitKey(0)

# Subsample the skeleton points to get the seeds
points = get_points_central_axis(skeleton)
seeds = points # Subsample every 10 point

# Step 1: Calculate the Sobel gradients Gx and Gy for the road lines
Gx = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)  # Gradient in x direction
Gy = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)  # Gradient in y direction

normals_image = np.zeros_like(central_axis) # Start with the road image to draw the normals
for seed in seeds:
    x, y = seed
    # Get the normal direction at this point
    d1, d2 = get_normal_direction(Gx, Gy, x, y)
    print(d1,d2)
    if(d1!=0 and d2!=0):
        current_x=x
        current_y=y
        # Move forward along the normal direction until we hit an edge or max distance
        while True:
            # Check if we're out of bounds
            if current_x < 0 or current_x >= Gx.shape[0] or current_y < 0 or current_y >= Gx.shape[1]:
                break

            # Check if the current point is an edge in the edges map
            if edges_map[int(current_x), int(current_y)] > 0:
                break  # Stop when we hit an edge
            normals_image[int(current_x),int(current_y)]=255
            current_x += d1 
            current_y += d2 
        if 0 <= current_x < normals_image.shape[0] and 0 <= current_y < normals_image.shape[1]:
            cv2.circle(normals_image, (int(current_y), int(current_x)),1 , 255, -1)


        current_x=x
        current_y=y
        while True:
            # Check if we're out of bounds
            if current_x < 0 or current_x >= Gx.shape[0] or current_y < 0 or current_y >= Gx.shape[1]:
                break

            # Check if the current point is an edge in the edges map
            if edges_map[int(current_x), int(current_y)] > 0:
                break  # Stop when we hit an edge
            normals_image[int(current_x),int(current_y)]=255
            current_x -= d1 
            current_y -= d2 
        if 0 <= current_x < normals_image.shape[0] and 0 <= current_y < normals_image.shape[1]:
            cv2.circle(normals_image, (int(current_y), int(current_x)), 1, 255, -1)
    


# Step 4: Show the result
cv2.imshow("Normal Vectors", normals_image)
cv2.waitKey(0)

overlay=road.copy()
overlay[normals_image==255]=[0,0,255]
cv2.imshow("smoothed", overlay)
cv2.waitKey(0)

cv2.destroyAllWindows()

