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
    magnitude = np.sqrt(Gx[x, y]**2 + Gy[x, y]**2)
    d1 = Gy[x, y]/magnitude  # Perpendicular to Gy (invert the vertical component)
    d2 = Gx[x, y]/magnitude# The horizontal component stays the same
    return (d1, d2)


# Load the central axis image
central_axis = cv2.imread('images/axe0.png', cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Invert the image
# Convert the image to a binary boolean array to skeletonize it
binary_bool = central_axis > 0
skeleton = morphology.skeletonize(binary_bool)
skeleton = np.uint8(skeleton) * 255
cv2.imshow("Road", skeleton)
cv2.waitKey(0)
# Load the road image
road = cv2.imread('images/route0.png')
road=cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
road = cv2.bilateralFilter(road, 9, 100, 100)
# Convertir l'image floutée de BGR à LAB
lab_image = cv2.cvtColor(road, cv2.COLOR_BGR2LAB)
# Séparer les canaux L, A, et B
L, road_gray, B = cv2.split(lab_image)
road_gray=((road_gray - road_gray.min()) / (road_gray.max() - road_gray.min()) * 255).astype(np.uint8)

# Appliquer le filtre Canny pour obtenir les contours
edges_map =cv2.Canny(road_gray, 25, 125)
display_image("Edges", edges_map)

# Subsample the skeleton points to get the seeds
points = get_points_central_axis(skeleton)
seeds = points # Subsample every 10 point

# Step 1: Calculate the Sobel gradients Gx and Gy for the road lines
Gx = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)  # Gradient in x direction
Gy = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)  # Gradient in y direction

normals_image = np.zeros_like(central_axis) # Start with the road image to draw the normals
#On stock la distance moyenne à l'axe ! 
distance=0
for seed in seeds:
    x, y = seed
    # Get the normal direction at this point
    d1, d2 = get_normal_direction(Gx, Gy, x, y)
    if(d1!=0 or d2!=0):
        current_x=x
        current_y=y
        # Move forward along the normal direction until we hit an edge or max distance
        while True:
            # Check if we're out of bounds
            if current_x < 0 or current_x >= Gx.shape[0] or current_y < 0 or current_y >= Gx.shape[1]:
                break
             # Check 8 connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = current_x + dx, current_y+ dy
                    if nx >= 0 and nx < edges_map.shape[0] and ny >= 0 and ny < edges_map.shape[1]:
                        if(distance - (abs(x -nx) + abs(y -ny))>0):
                            normals_image[int(nx), int(ny)] = 255
            #les contours
            if edges_map[int(current_x),int(current_y)] > 0:
                print("yo")
                if(distance !=0):
                    current_dist= abs(x -current_x) + abs(y -current_y)
                    if( abs(distance - current_dist)<10):
                        normals_image[int(current_x),int(current_y)]=255
                        distance = (current_dist + distance)/2
                else:
                    distance= abs(current_y -y) + abs(current_x -x)
                    normals_image[int(current_x),int(current_y)]=255
                break
            current_x += d1 
            current_y += d2 

        current_x=x
        current_y=y
        while True:
            # Check if we're out of bounds
            if current_x < 0 or current_x >= Gx.shape[0] or current_y < 0 or current_y >= Gx.shape[1]:
                break
            # Check 8 connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = current_x + dx, current_y+ dy
                    if nx >= 0 and nx < edges_map.shape[0] and ny >= 0 and ny < edges_map.shape[1]:
                        if(distance - (abs(x -nx) + abs(y -ny))>0):
                            normals_image[int(nx), int(ny)] = 255
            normals_image[int(current_x),int(current_y)]=255
            #les contours
            if edges_map[int(current_x),int(current_y)] > 0:
                if(distance !=0):
                    current_dist= abs(x -current_x) + abs(y -current_y)
                    if( abs(distance - current_dist)<10):
                        normals_image[int(current_x),int(current_y)]=255
                        distance = (current_dist + distance)/2
                else:
                    distance= abs(current_y -y) + abs(current_x -x)
                    normals_image[int(current_x),int(current_y)]=255
                break
            current_x -= d1 
            current_y -= d2 
    

# Apply morphological closing to fill the gaps
kernel = np.ones((3, 3), np.uint8)
open = cv2.morphologyEx(normals_image, cv2.MORPH_OPEN, kernel, iterations=3)

# Smooth the edges by applying median filter
smoothed = cv2.medianBlur(open, 9)
# Step 4: Show the result
cv2.imshow("Normal Vectors", smoothed)
cv2.waitKey(0)

overlay=road.copy()
overlay[smoothed==255]=[0,0,255]
cv2.imshow("smoothed", overlay)
cv2.waitKey(0)

cv2.destroyAllWindows()

