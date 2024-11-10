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

# Load the central axis image
central_axis = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
central_axis = scale_image(central_axis)
central_axis = cv2.bitwise_not(central_axis) # Invert the image

# Skeletonize the central axis image
skeleton, points = skeletonize_image(central_axis)
skeleton_copy = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

# Sobel filter to compute the gradient
sobel_x = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=7)
sobel_y = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=7)

# Draw the normals
step = 5 # Draw normals every 5 points
line_length = 15

for k in range(0, len(points), step):
    # Extract k-th point of the central axis
    (i, j) = points[k]
    grad_x = sobel_x[i, j]
    grad_y = sobel_y[i, j]

    if grad_x != 0 or grad_y != 0:
        # Normalize the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude

        # Compute the line coordinates
        x1 = j
        y1 = i
        x2 = int(j + line_length * norm_x)
        y2 = int(i + line_length * norm_y)

        # Compute the line coordinates in the opposite direction
        x3 = int(j - line_length * norm_x)
        y3 = int(i - line_length * norm_y)

        # Draw the normal lines
        cv2.circle(skeleton_copy, (x1, y1), 3, (255, 0, 0), -1)
        cv2.line(skeleton_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(skeleton_copy, (x1, y1), (x3, y3), (0, 255, 0), 2)
        

display_image("Normals", skeleton_copy)
