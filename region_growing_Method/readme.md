# mask.py
Central Axis Skeletonization and Interactive Dilation: This script allows the extraction of the central axis from road images using skeletonization. It provides an interactive interface to adjust the dilation kernel size, refining the road mask for better visualization.
```bash
python3 .\Region_Growing\mask.py -img 0
```

# region_growing.py
This Python script performs road segmentation using a region-growing algorithm. It allows the user to interactively select seed points from the image and segments the image based on intensity similarity. The script supports adjusting the filtering size for preprocessing and the threshold for region-growing via command-line arguments.
```bash
python Region_Growing.py <threshold_value> <filter_size>
```
#road_segmentation_proba.py
This Python script performs advanced road segmentation using a combination of region-growing and morphological techniques. The process starts by masking the road image using a central axis, followed by region-growing segmentation based on intensity similarity from selected seed points. Afterward, post-processing steps (such as morphological closing and median filtering) refine the segmentation results to improve accuracy.

```bash
python3 .\Region_Growing\road_segmentation_Proba.py
```

# road_segmentation_RG.py
This script performs road image segmentation using region growing based on a skeletonized central axis, followed by post-processing steps such as morphological closing and smoothing to enhance the segmentation quality
```bash
python3 .\Region_Growing\road_segmentation_RG.py -img 0 -thresh 15
```