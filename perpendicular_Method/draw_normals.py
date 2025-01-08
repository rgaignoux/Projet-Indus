import cv2
import numpy as np
import sys
import argparse
import utils
import math

def angle_between_vectors(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2 + 1e-8)
    return np.arccos(np.clip(cos_theta, -1, 1))

def correct_normal_orientation(normals):
    for idx in range(1, len(normals)):
        _, (prev_norm_x, prev_norm_y) = normals[idx - 1]
        (i, j), (curr_norm_x, curr_norm_y) = normals[idx]

        # Dot product between current and previous
        dot_product = prev_norm_x * curr_norm_x + prev_norm_y * curr_norm_y

        # If dot product < 0, angle between the 2 vectors > 90°, we invert the normal
        if dot_product < 0:
            normals[idx] = ((i, j), (-curr_norm_x, -curr_norm_y))

    return normals

def remove_normals_outliers(normals, angle_threshold = np.pi / 4):

    corrected_normals = []

    for idx in range(len(normals)):
        (i, j), (norm_x, norm_y) = normals[idx]

        # Extract 2 normal neighbors (previous and next ones)
        neighbor_normals = []
        for offset in [-1, 1]:
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(normals):
                _, (neighbor_x, neighbor_y) = normals[neighbor_idx]
                neighbor_normals.append((neighbor_x, neighbor_y))
                
        # Compute average delta x and delta y
        if neighbor_normals:
            avg_x = np.mean([n[0] for n in neighbor_normals])
            avg_y = np.mean([n[1] for n in neighbor_normals])
            avg_normal = (avg_x, avg_y)

            # Angle between the 2 vectors (using the dot product)
            angle = angle_between_vectors((norm_x, norm_y), avg_normal)

            # Replace outlier by average
            if angle >= angle_threshold:
                corrected_normals.append(((i, j), avg_normal))
            else:
                corrected_normals.append(((i, j), (norm_x, norm_y)))
        
        else:
            corrected_normals.append(((i, j), (norm_x, norm_y)))

    return corrected_normals

def extract_normals(central_axis):
    # Skeletonize the central axis image
    skeleton, points = utils.skeletonize_image(central_axis)
    points = utils.sort_points(points)

    # Sobel filter to compute the gradient
    sobel_x = cv2.Sobel(central_axis, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(central_axis, cv2.CV_64F, 0, 1, ksize=5)

    # Compute normals
    normals = []
    for k in range(0, len(points)):
        (i, j) = points[k]
        grad_x = sobel_x[i, j]
        grad_y = sobel_y[i, j]

        if grad_x != 0 or grad_y != 0:
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            norm_x = grad_x / magnitude
            norm_y = grad_y / magnitude
            normals.append(((i, j), (norm_x, norm_y)))
        else:
            normals.append(((i, j), (0, 0)))

    # Correct orientation if needed
    normals = correct_normal_orientation(normals)

    # Remove "outliers"
    normals = remove_normals_outliers(normals)

    # Transform to dict
    normals_dict = {key: value for key, value in normals}

    return normals_dict, points

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str)
    args = parser.parse_args()

    central_axis_path = str(args.path)

    # Load the central axis image
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
    central_axis = utils.scale_image(central_axis)
    central_axis = cv2.bitwise_not(central_axis) # Invert the image
    skeleton, points = utils.skeletonize_image(central_axis)
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # Compute normals
    normals, _ = extract_normals(central_axis)

    line_length = 15
    for (i, j), (norm_x, norm_y) in normals.items():
        if norm_x != 0 or norm_y != 0:
            x1 = j
            y1 = i
            x2 = int(j + line_length * norm_x)
            y2 = int(i + line_length * norm_y)
            x3 = int(j - line_length * norm_x)
            y3 = int(i - line_length * norm_y)

            cv2.circle(skeleton_rgb, (x1, y1), 3, (255, 0, 0), -1)
            cv2.line(skeleton_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(skeleton_rgb, (x1, y1), (x3, y3), (0, 255, 0), 2)

    utils.display_image("Normals", skeleton_rgb)