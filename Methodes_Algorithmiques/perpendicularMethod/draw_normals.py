import cv2
import numpy as np
import sys
import argparse
import utils
import math
import os
import json

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str)
    args = parser.parse_args()

    central_axis_path = str(args.path)
    central_axis = cv2.imread(central_axis_path, cv2.IMREAD_GRAYSCALE)
    pic_min_x, pic_min_y, pic_max_x, pic_max_y = utils.get_min_max(central_axis_path)
    skeleton, points = utils.skeletonize_image(central_axis)
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    json_file = 'Methodes_Algorithmiques/perpendicularMethod/Data/filaires-projection.json'
    routes = utils.load_filaires(json_file, pic_min_x, pic_min_y, pic_max_x, pic_max_y)

    normals = {}

    for route in routes:
        for line in route:
            projected_line = utils.project_on_image(line, pic_min_x, pic_min_y, pic_max_x, pic_max_y, central_axis.shape[1], central_axis.shape[0])
            
            for i in range(len(projected_line) - 1):
                vec = np.array(projected_line[i]) - np.array(projected_line[i + 1])
                normal = vec[1], -vec[0]
                norm_x, norm_y = normal / np.linalg.norm(normal)

                temp = np.zeros((central_axis.shape[0], central_axis.shape[1]))
                cv2.line(temp, projected_line[i], projected_line[i + 1], 255, 1)

                indexes = np.argwhere(temp > 0)

                for (i, j) in indexes:
                    normals[(i, j)] = (norm_x, norm_y)
    

    line_length = 15
    c = 0
    for (i, j), (norm_x, norm_y) in normals.items():
        c += 1
        if c == 3:
            if norm_x != 0 or norm_y != 0:
                x1 = j
                y1 = i
                x2 = int(j + line_length * norm_x)
                y2 = int(i + line_length * norm_y)
                x3 = int(j - line_length * norm_x)
                y3 = int(i - line_length * norm_y)

                cv2.circle(skeleton_rgb, (x1, y1), 1, (255, 0, 0), -1)
                cv2.line(skeleton_rgb, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.line(skeleton_rgb, (x1, y1), (x3, y3), (0, 255, 0), 1)

            c = 0

    utils.display_image("Normals", skeleton_rgb, scale=True)
    filename = os.path.splitext(os.path.basename(central_axis_path))[0]
    cv2.imwrite(f".\\Methodes_Algorithmiques\\perpendicular_Method_realOrtho\\results\\normals_{filename}.png", skeleton_rgb)