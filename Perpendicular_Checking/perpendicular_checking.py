import cv2
import numpy as np
import sys
from skimage import morphology

def display_image(window_name, img):
    """
    Display an image in a window with the given name.
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def resize_image(img, width, height):
    """
    Resize the image to the given width and height.
    """
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    
    return img

def scale_image(img, scale_percent=65):
    """
    Scale the image by the given scale percentage.
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return img

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


central_axis_path = sys.argv[1]
road_path = sys.argv[2]

# Load the road image
road = cv2.imread(road_path, cv2.IMREAD_COLOR)
road = scale_image(road)

# Load the central axis image
central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
central_axis = resize_image(central_axis, road.shape[1], road.shape[0])
central_axis = cv2.bitwise_not(central_axis) # Invert the image

# Sobel filter to compute the gradient
sobel_x = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)
sobel_y = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)

# Skeletonize the central axis image
skeleton, points = skeletonize_image(central_axis)

# Bilateral filtering
road = cv2.bilateralFilter(road, 9, 100, 100)

# Canny edge detection
edges = cv2.Canny(road, 75, 125)
display_image("Edges", edges)

# Overlay the skeleton on the edges image
edges_copy = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges_copy[skeleton > 0] = [0, 255, 0]
display_image("Edges with Central Axis", edges_copy)

# Find the road edges
road_edges = np.zeros(road.shape[:2], dtype=np.uint8)
minmax = range(3, 15) # Min and max distances to check for edges

for k in range(0, len(points)):
    # Extract k-th point of the central axis
    (i, j) = points[k]
    grad_x = sobel_x[i, j]
    grad_y = sobel_y[i, j]

    if grad_x != 0 or grad_y != 0:
        # Normalize the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude

        # Check the edges in the normal direction
        for n in minmax:
            x2 = int(j + n * norm_x)
            y2 = int(i + n * norm_y)

            if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                if edges[y2, x2] == 0:
                    road_edges[y2, x2] = 255

                elif edges[y2, x2] > 0:
                    road_edges[y2, x2] = 255

                    # Check 8 connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y2 + dy, x2 + dx
                            if ny >= 0 and ny < edges.shape[0] and nx >= 0 and nx < edges.shape[1]:
                                if edges[ny, nx] > 0:
                                    road_edges[ny, nx] = 255
                    break

        # Check the edges in the opposite direction
        for n in minmax:
            x3 = int(j - n * norm_x)
            y3 = int(i - n * norm_y)

            if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                if edges[y3, x3] == 0:
                    road_edges[y3, x3] = 255

                elif edges[y3, x3] > 0:
                    road_edges[y3, x3] = 255

                    # Check 8 connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y3 + dy, x3 + dx
                            if ny >= 0 and ny < edges.shape[0] and nx >= 0 and nx < edges.shape[1]:
                                if edges[ny, nx] > 0:
                                    road_edges[ny, nx] = 255
                    break

# Overlay the road edges on the road image
road_copy = road.copy()
road_copy[road_edges > 0] = [0, 0, 255]
road_copy[central_axis > 0] = [0, 255, 0]
display_image("Road Edges", road_copy)
