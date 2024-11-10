import cv2
import numpy as np
from utils import *
from region_growing import *
from skimage import io, morphology

def skeletonize_image(img):
    """
    Skeletonize the given image.
    """
    # Convert the image to a binary boolean array to skeletonize it
    binary_bool = img > 0

    # Skeletonize the image
    skeleton = morphology.skeletonize(binary_bool)
    skeleton = np.uint8(skeleton) * 255

    # Exract the points from the skeleton
    points = get_points_central_axis(skeleton)

    return skeleton, points

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
    Computes the normal direction at a given point based on the Sobel gradient.
    
    Parameters:
    - Gx: Gradient in the x-direction (Sobel X).
    - Gy: Gradient in the y-direction (Sobel Y).
    - x: x-coordinate of the point.
    - y: y-coordinate of the point.
    
    Returns:
    - (d1, d2): The normal direction components (perpendicular to the gradient).
    """
    magnitude = np.sqrt(Gx[x, y]**2 + Gy[x, y]**2)
    
    # Perpendicular to the gradient components
    d1 = Gy[x, y] / magnitude
    d2 = Gx[x, y] / magnitude
    
    return (d1, d2)


# Load and preprocess the central axis image
central_axis = cv2.imread('images/axe0.png', cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis)
central_axis = cv2.bitwise_not(central_axis)  # Invert the image

# Skeletonize the central axis
skeleton, seeds  = skeletonize_image(central_axis)

# Load and preprocess the road image
road = cv2.imread('images/route0.png')
road=cv2.resize(road, (central_axis.shape[1], central_axis.shape[0]), interpolation=cv2.INTER_NEAREST)
road = cv2.bilateralFilter(road, 9, 100, 100)

# Convert the road image to the LAB color space and extract the L and B channels
lab_image = cv2.cvtColor(road, cv2.COLOR_BGR2LAB)
L, road_gray, B = cv2.split(lab_image)

# Normalize the road grayscale image
road_gray = ((road_gray - road_gray.min()) / (road_gray.max() - road_gray.min()) * 255).astype(np.uint8)


# Apply Canny edge detection
edges_map = cv2.Canny(road_gray, 25, 125)

# Step 1: Calculate the Sobel gradients Gx and Gy for the road lines
Gx = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)  # Gradient in x direction
Gy = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)  # Gradient in y direction

# Initialize an image to store the normal vectors
normals_image = np.zeros_like(central_axis)
distance = 0  # Initialize the average distance

# Process each seed point to trace normals
for seed in seeds:
    x, y = seed
    d1, d2 = get_normal_direction(Gx, Gy, x, y)

    if(d1!=0 or d2!=0): #case where here is no gradient
        current_x, current_y = x, y
         # Move forward along the normal direction until we hit an edge or reach maximum distance
        while True:
            if current_x < 0 or current_x >= Gx.shape[0] or current_y < 0 or current_y >= Gx.shape[1]:
                break

            # Check the 8-connected neighbors around the current point  PEUT ETRE A MODIFIER EN FCT DE L ALGO FINAL
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = current_x + dx, current_y+ dy
                    if nx >= 0 and nx < edges_map.shape[0] and ny >= 0 and ny < edges_map.shape[1]:
                        if(distance - (abs(x -nx) + abs(y -ny))>0): #A MODIFIER ?
                            normals_image[int(nx), int(ny)] = 255
            # Check if we reached an edge (based on the Canny edges)
            if edges_map[int(current_x),int(current_y)] > 0:
                if(distance !=0):
                    current_dist= abs(x -current_x) + abs(y -current_y)
                    if(abs(distance - current_dist)<10):
                        normals_image[int(current_x),int(current_y)]=255
                        distance = (current_dist + distance)/2
                else:
                    distance= abs(current_y -y) + abs(current_x -x)
                    normals_image[int(current_x),int(current_y)]=255
                break
            current_x += d1 
            current_y += d2 

        # Repeat the same process in the opposite direction
        current_x, current_y = x, y
        while True:
            if current_x < 0 or current_x >= Gx.shape[0] or current_y < 0 or current_y >= Gx.shape[1]:
                break
            # Check the 8-connected neighbors around the current point
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = current_x + dx, current_y+ dy
                    if nx >= 0 and nx < edges_map.shape[0] and ny >= 0 and ny < edges_map.shape[1]:
                        if(distance - (abs(x -nx) + abs(y -ny))>0):
                            normals_image[int(nx), int(ny)] = 255
            normals_image[int(current_x),int(current_y)]=255
            # Check if we reached an edge
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
    

# Step 2: Apply morphological operations to fill gaps
kernel = np.ones((3, 3), np.uint8)
opened_normals = cv2.morphologyEx(normals_image, cv2.MORPH_OPEN, kernel, iterations=3)

# Step 3: Smooth the edges using a median filter
smoothed_normals = cv2.medianBlur(opened_normals, 9)

        # --- Display Results ---
# 1. Visualize the central axis on the road (green color)
road_with_axis = road.copy()
for (x, y) in seeds:
    road_with_axis[x, y] = [0, 255, 0]  # Mark central axis with green color
cv2.imshow("Road with Central Axis (Green)", road_with_axis)
cv2.waitKey(0)

# 2. Visualize Canny edges
cv2.imshow("Canny Edges", edges_map)
cv2.waitKey(0)

# 3. Overlay the normal vectors on the road image (in red) before post-processing
road_before_post = road.copy()
road_before_post[normals_image == 255] = [0, 0, 255]  # Red color for the normal vectors (before post-processing)

# 4. Overlay the normal vectors on the road image (in red) after post-processing
road_after_post = road.copy()
road_after_post[smoothed_normals == 255] = [0, 0, 255]  # Red color for the normal vectors (after post-processing)

# 5. Display both before and after post-processing images side by side
combined_image = np.hstack((road_before_post, road_after_post))  # Concatenate both images horizontally
cv2.imshow("Road with Normal Vectors (Before and After Post-processing)", combined_image)
cv2.waitKey(0)

# 5. Overlay the normal vectors on the road image (in red)
overlay = road.copy()
overlay[smoothed_normals == 255] = [0, 0, 255]  # Red color for the normal vectors
cv2.imshow("Road with Normal Vectors (Red)", overlay)

cv2.destroyAllWindows()

