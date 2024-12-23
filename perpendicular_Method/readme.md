# visualisation_road_corners.py
This function detects the road contours starting from the central axis, identifying them by drawing a red point when a contour is detected by the Canny edge detection.

```bash
.\perpendicular_Method\segmentation_corners_road.py -img 0   -min 75 -max 125
```

# road_segmentation_perpendicular.py
This script detects and visualizes the normal vectors and road contours based on the central axis. It starts from the central axis and traces the perpendicular normal direction, identifying edges using the Canny edge detection method. After identifying the road contours, the script overlays the normal vectors on the road image, both before and after post-processing.


```bash
 python3 .\perpendicular_Method\road_segmentation_Perpendiculaire.py -img 0 -min 75 -max 125
 ```



# average_width.py
This Python script performs road edge detection using various computer vision techniques such as Sobel filtering, Canny edge detection, and skeletonization. It also calculates the average width of the road edges based on a given central axis image. (for each point detected as a corner)

```bash
python3 .\perpendicular_Method\average_width.py -img 5 -min 75 -max 125
```
