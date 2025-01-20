import cv2
import numpy as np
import json
import utils_forOrtho as utils
import argparse
import math
import glob
import os

def remove_outliers_and_compute_mean(widths_around, threshold = 1.75):
    if not widths_around:
        return 0

    # Remove outliers using z-score method
    average = np.mean(widths_around)
    if np.std(widths_around) != 0:
        z_scores = np.abs((widths_around - average) / np.std(widths_around))
        widths_around = [w for w, z in zip(widths_around, z_scores) if z < threshold]

    # Update the average width
    if widths_around:
        average = np.mean(widths_around)
        #np.quantile(widths_around, 0.75)

    return int(average)

'python .\Methodes_Algorithmiques/perpendicular_Method_realOrtho\segmentation_moyennesegment.py -dir=".\Methodes_Algorithmiques\images\ImagesRennes\"'
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str) # Directory path containing the images 
args = parser.parse_args()
directory_path = args.dir
minmax = range(1, 50) # Min and max distances to check for edges
json_file = 'Methodes_Algorithmiques/perpendicular_Method_realOrtho/Data/filaires-projection.json'
display = 1

# Images paths
road_paths = [os.path.basename(file) for file in glob.glob(f"{directory_path}//*.png")]
axes_paths = glob.glob(f"{directory_path}//axe*.png")
road_paths = glob.glob(f"{directory_path}//road*.png")

for (central_axis_path, road_path) in zip(axes_paths, road_paths):
    print("Processing image...")

    # Load the road image
    road = cv2.imread(road_path, cv2.IMREAD_COLOR)

    # Load the central axis image
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)

    # Get min_x, min_y, max_x, max_y
    pic_min_x, pic_min_y, pic_max_x, pic_max_y = utils.get_min_max(central_axis_path)

    # Filter out the white pixels
    hsv_image = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    hsv_image[mask > 0] = [0, 0, 160]
    road2 = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Gaussian filtering
    road_blurred = cv2.bilateralFilter(road2, 9, 150, 150)

    # Canny edge detection
    edges = cv2.Canny(road_blurred, 75, 100)

    # Create the segmentation mask
    segmentation_mask = np.zeros(road.shape[:2])

    # Loop on the different roads in the image
    routes = utils.load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y)

    for route in routes:
        for line in route:

            projected_line = utils.project_on_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, road.shape[1], road.shape[0]) #vraiment road.shape?
            
            for i in range(len(projected_line) - 1):
                # Compute normal vector
                vec = np.array(projected_line[i]) - np.array(projected_line[i + 1])
                normal = vec[1], -vec[0]
                norm_x, norm_y = normal / np.linalg.norm(normal)

                # Draw the line in a temp image
                temp = np.zeros((central_axis.shape[0], central_axis.shape[1]))
                cv2.line(temp, projected_line[i], projected_line[i + 1], 255, 1)

                # Get (i, j) coordinates of all points on the line
                points = np.argwhere(temp > 0) 

                widths1 = []
                widths2 = []

                for k in range(0, len(points)):
                    (i, j) = points[k]

                    width1 = 0
                    width2 = 0
                    
                    # Check the edges in the normal direction
                    for n in minmax:
                        x2 = int(j + n * norm_x)
                        y2 = int(i + n * norm_y)

                        if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                            width1 += 1
                            if edges[y2, x2] > 0:
                                break

                    # Check the edges in the opposite direction
                    for n in minmax:
                        x3 = int(j - n * norm_x)
                        y3 = int(i - n * norm_y)

                        if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                            width2 += 1
                            if edges[y3, x3] > 0:
                                break

                    widths1.append(width1)
                    widths2.append(width2)


                # Compute the average widths
                average_width1 = remove_outliers_and_compute_mean(widths1)
                average_width2 = remove_outliers_and_compute_mean(widths2)
        
                # Draw the segmentation
                for k in range(0, len(points)):
                    (i, j) = points[k]

                    # Flag in the normal direction
                    for n in range(average_width1): 
                        x2 = int(j + n* norm_x )
                        y2 = int(i + n* norm_y )

                        if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                            segmentation_mask[y2, x2] = 1
                        
                    # Flag in the opposite direction
                    for n in range(average_width2):
                        x3 = int(j - n* norm_x )
                        y3 = int(i - n* norm_y )

                        if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                            segmentation_mask[y3, x3] = 1


    # Post process segmentation
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
    filename = os.path.splitext(os.path.basename(road_path))[0]
    cv2.imwrite(f"Methodes_Algorithmiques/perpendicular_Method_realOrtho//results//segm_{filename}.png", segmentation_mask)

    # Overlay the segmentation on the road image
    result = road.copy()
    result[central_axis >= 1] = np.array([0, 0, 0])
    result[segmentation_mask >= 1] = (0.4 * np.array([0, 255, 255]) + 0.6 * result[segmentation_mask >= 1]).astype(np.uint8)
    
    edges_with_axis = np.dstack((edges, edges, edges))
    edges_with_axis[central_axis >= 1] = (255, 0, 0)
    stacked = np.hstack((edges_with_axis, result))
    cv2.imwrite(f"Methodes_Algorithmiques/perpendicular_Method_realOrtho//results//result_stacked_{filename}.png", stacked)
    
    utils.display_image("Result", stacked, scale = True)
       