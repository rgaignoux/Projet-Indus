import cv2
import numpy as np
import utils
import argparse
import math
import glob
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str) # Directory path containing the images 
args = parser.parse_args()
directory_path = args.dir
minmax = range(0, 50) # Min and max distances to check for edges
json_file = 'Methodes_Algorithmiques/perpendicularMethod/Data/filaires-projection.json' # Fichiers contenant les segments de routes
display = 0 # 1 : display images during the process ; 0 : don't display

# Images paths
axes_paths = glob.glob(f"{directory_path}//axe*.png")
road_paths = glob.glob(f"{directory_path}//road*.png")
segm_paths = glob.glob(f"{directory_path}//segm*.png")

for (central_axis_path, road_path, segm_path) in zip(axes_paths, road_paths, segm_paths):
    print("Processing image...")

    # Load the road image
    road = cv2.imread(road_path, cv2.IMREAD_COLOR)

    # Load the central axis image
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)

    # Load the segmentation image
    segmentation_mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

    # Get min_x, min_y, max_x, max_y in the image filename
    pic_min_x, pic_min_y, pic_max_x, pic_max_y = utils.get_min_max(central_axis_path)

    # Loop on the different roads in the image
    routes = utils.load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y)

    # New segmentation
    new_segmentation_mask = np.zeros(road.shape[:2])

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

                    width1 = minmax[0]
                    width2 = minmax[0]
                    
                    # Check the edges in the normal direction
                    for n in minmax:
                        x2 = int(j + n * norm_x)
                        y2 = int(i + n * norm_y)

                        if x2 >= 0 and x2 < segmentation_mask.shape[1] and y2 >= 0 and y2 < segmentation_mask.shape[0]:
                            width1 += 1
                            if segmentation_mask[y2, x2] == 0:
                                break

                    # Check the edges in the opposite direction
                    for n in minmax:
                        x3 = int(j - n * norm_x)
                        y3 = int(i - n * norm_y)

                        if x3 >= 0 and x3 < segmentation_mask.shape[1] and y3 >= 0 and y3 < segmentation_mask.shape[0]:
                            width2 += 1
                            if segmentation_mask[y3, x3] == 0:
                                break

                    widths1.append(width1)
                    widths2.append(width2)

                # Compute the average widths
                average_width1 = utils.remove_outliers_and_compute_mean(widths1)
                average_width2 = utils.remove_outliers_and_compute_mean(widths2)
        
                # Draw the segmentation
                for k in range(0, len(points)):
                    (i, j) = points[k]

                    # Flag in the normal direction
                    for n in range(average_width1): 
                        x2 = int(j + n* norm_x )
                        y2 = int(i + n* norm_y )

                        if x2 >= 0 and x2 < segmentation_mask.shape[1] and y2 >= 0 and y2 < segmentation_mask.shape[0]:
                            new_segmentation_mask[y2, x2] = 1
                        
                    # Flag in the opposite direction
                    for n in range(average_width2):
                        x3 = int(j - n* norm_x )
                        y3 = int(i - n* norm_y )

                        if x3 >= 0 and x3 < segmentation_mask.shape[1] and y3 >= 0 and y3 < segmentation_mask.shape[0]:
                            new_segmentation_mask[y3, x3] = 1


    # Post process segmentation
    new_segmentation_mask = cv2.morphologyEx(new_segmentation_mask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=1)
    new_segmentation_mask = (new_segmentation_mask * 255).astype(np.uint8)
    filename = os.path.splitext(os.path.basename(road_path))[0]
    cv2.imwrite(f"PostProcess/new_masks/segm_{filename}.png", new_segmentation_mask)
    
    # Overlay the segmentation on the road image
    result = road.copy()
    central_axis_dilate = cv2.dilate(central_axis, np.ones((3, 3)))
    result[central_axis_dilate >= 1] = np.array([255, 0, 0])
    result[new_segmentation_mask >= 1] = (0.5 * np.array([0, 0, 255]) + 0.5 * result[new_segmentation_mask >= 1]).astype(np.uint8)
    cv2.imwrite(f"PostProcess/results/overlay_{filename}.png", result)
    
    #utils.display_image("Result", result, scale = True)