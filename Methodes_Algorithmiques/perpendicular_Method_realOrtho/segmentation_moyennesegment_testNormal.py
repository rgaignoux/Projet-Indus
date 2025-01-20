import cv2
import numpy as np
import json
import utils_forOrtho as utils
import argparse
import math
import glob
import os
from draw_normals import extract_normals


'python3 .\Methodes_Algorithmiques/perpendicular_Method_realOrtho\segmentation_moyennesegment.py -dir="Methodes_Algorithmiques/perpendicular_Method_realOrtho/images_to_segment"'
def remove_outliers(widths_around, average, threshold = 0.5):
    # Remove outliers using z-score method
    if np.std(widths_around) != 0:
        z_scores = np.abs((widths_around - average) / np.std(widths_around))
        widths_around = [w for w, z in zip(widths_around, z_scores) if z < threshold]

    # Update the average width
    if widths_around:
        average = np.mean(widths_around)

    return average
# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str) # Directory path containing the images
parser.add_argument('-min', type=int, default=1) # Min range for edge detection
parser.add_argument('-max', type=int, default=30) # Max range for edge detection
parser.add_argument('-display', type=int, default=0) # 0 : don't display images, 1 : display images 

args = parser.parse_args()
minmax = range(args.min, args.max) # Min and max distances to check for edges
display = args.display

# Images paths
directory_path = args.dir
road_paths = [os.path.basename(file) for file in glob.glob(f"{directory_path}//*.png")]
for road_path in  road_paths:
    print("Processing image...")

    # Load the road image
    road, pic_min_x, pic_min_y, pic_max_x, pic_max_y = utils.load_orthophoto(directory_path+"/", road_path)
    # Filter out the white pixels
    hsv_image = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    hsv_image[mask > 0] = [0, 0, 160]
    road2 = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Gaussian filtering
    road_blurred = cv2.bilateralFilter(road2, 9, 150, 150)

    # Canny edge detection
    edges = cv2.Canny(road_blurred, 100, 125)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)


    # Create the segmentation mask
    segmentation_mask = np.zeros(road.shape[:2])

    #Loop on the different roads in the image
    json_file = 'Methodes_Algorithmiques/perpendicular_Method_realOrtho/Data/filaires-projection.json'
    routes = utils.load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y)

    for route in routes:
        central_axis= np.zeros((road.shape[0], road.shape[1]))
        for line in route:
            projected_line = utils.project_on_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, road.shape[1], road.shape[0]) #vraiment road.shape?
            for i in range(len(projected_line) - 1):
                cv2.line(central_axis, projected_line[i], projected_line[i + 1], 255, 2)

        normals, points = extract_normals(central_axis)
        mean_norm_x = np.mean([norm_x for (norm_x, _) in normals.values()])
        mean_norm_y = np.mean([norm_y for (_, norm_y) in normals.values()])
        widths1 = []
        widths2 = []

        for k in range(0, len(points)):
            # Extract k-th point of the central axis
            (i, j) = points[k]

            width1 = 0
            width2 = 0

            (norm_x, norm_y) = normals[(i, j)]
            norm_x=mean_norm_x
            norm_y=mean_norm_y
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
        average_widths1 = np.mean(widths1)
        average_widths2 = np.mean(widths2)
        widths1=remove_outliers(widths1, average_widths1)
        widths2=remove_outliers(widths2, average_widths2)
        
        for pos in points:

            (i, j) = pos

            (norm_x, norm_y) = normals[(i, j)]
            norm_x=mean_norm_x
            norm_y=mean_norm_y
            # Flag in the normal direction
            for n in range(width1): 
                x2 = int(j + n* norm_x )
                y2 = int(i + n* norm_y )

                if x2 >= 0 and x2 < edges.shape[1] and y2 >= 0 and y2 < edges.shape[0]:
                    segmentation_mask[y2, x2] = 1
                
            # Flag in the opposite direction
            for n in range(width2):
                x3 = int(j - n* norm_x )
                y3 = int(i - n* norm_y )

                if x3 >= 0 and x3 < edges.shape[1] and y3 >= 0 and y3 < edges.shape[0]:
                    segmentation_mask[y3, x3] = 1


    # Post process segmentation
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
    filename = os.path.splitext(os.path.basename(road_path))[0]
    cv2.imwrite(f"Methodes_Algorithmiques/perpendicular_Method_realOrtho//results//segm_{filename}_mean.png", segmentation_mask)
    if display == 1:
        utils.display_image("Segmentation",segmentation_mask)

    # Overlay the segmentation on the road image
    result = road.copy()
    result[central_axis >= 1] = np.array([0, 0, 0])
    result[segmentation_mask >= 1] = (0.4 * np.array([0, 255, 255]) + 0.6 * result[segmentation_mask >= 1]).astype(np.uint8)
    cv2.imwrite(f"Methodes_Algorithmiques/perpendicular_Method_realOrtho//results//overlay_{filename}_mean.png", result)
    if display == 1:
        utils.display_image("Result", result)