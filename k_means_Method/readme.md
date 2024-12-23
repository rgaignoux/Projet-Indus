# Road Image Segmentation using K-means and Skeletonization: Kmeans.py
This Python script performs road segmentation from images using K-means clustering on the LAB color space and skeletonization. It processes a central axis (skeleton) and uses it to segment and highlight the road regions from the input images.
## Functionality
1. **K-means segmentation**: Segments the road and background using the A channel of the LAB color space.
2. **Skeletonization**: Analyzes the central axis of the road image for better mask refinement.
3. **Morphological Operations**: Applies morphological opening to refine the segmented road mask.
4. **Overlay visualization**: Displays the original image with the segmented road mask overlay.
## Requirements
- OpenCV (`cv2`)
- NumPy
- Scikit-image
- Matplotlib
## Usage

python Kmeans.py -img <image_number>
exemple:python .\k_means_Method\Kmeans.py -img 5


# Road Image Segmentation using Fuzzy C-means: Fuzzy_c_means.py
This Python script performs road segmentation from images using **Fuzzy C-means** clustering on the LAB color space and skeletonization. It processes a central axis (skeleton) and uses it to segment and highlight the road regions in the input images. This approach allows for a soft clustering of the road and background, where pixels can belong to multiple clusters with different degrees of membership.

## Functionality
1. **Fuzzy C-means segmentation**: Segments the road and background using the A channel of the LAB color space with Fuzzy C-means clustering.
2. **Skeletonization**: Extracts and analyzes the central axis of the road image to refine the segmentation mask.
3. **Morphological Operations**: Applies morphological opening to refine the segmented road mask.
4. **Overlay visualization**: Displays the original image with the segmented road mask overlay.

## How It Works
- **Fuzzy C-means Clustering**: The road image is first converted to the **LAB color space** to separate the lightness (L) from the color components (A and B). The `A` channel, which is most sensitive to color changes in road images (e.g., green and red), is used for segmentation. Fuzzy C-means is applied to this channel, where each pixel is assigned a membership to multiple clusters, and the pixel can belong to several clusters with different degrees of confidence.

- **Skeletonization**: The central axis (skeleton) of the road image is extracted from the pre-processed grayscale image. This skeleton helps to focus the segmentation on the road region by creating a precise mask based on the central line of the road.

- **Membership Calculation**: After performing the Fuzzy C-means clustering, the membership matrix `u` provides the degree of membership of each pixel in each cluster. The `u_max` matrix is used to assign each pixel to the cluster with the highest membership, which is then used for further processing.

- **Mask Refinement**: The segmented mask is refined by applying morphological operations, such as **opening**, to remove noise and fill small gaps. The resulting mask isolates the road from the background.

## Requirements
- OpenCV (`cv2`)
- NumPy
- Scikit-image
- Matplotlib
- Scikit-fuzzy (`fuzz`)

## How to Use
Run the script from the command line by specifying the image number you want to process. The script takes an image number as input and loads the corresponding road and central axis images.

python .\k_means_Method\Fuzzy_c_means.py -img <image_number>