import cv2
import numpy as np

# Global variables
seeds = []  # List to store seed positions
threshold = 15  # Default threshold value for region growing

def on_mouse_click(event, x, y, flags, param):
    """
    Mouse callback function to capture seed points when the user left-clicks on the image.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append((y, x))  # Append the coordinates (row, col) format
        # Draw a circle on the image where the seed is added
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Select Seeds", img_display)

def region_growing(image, seeds, threshold):
    """
    Perform region growing from given seed points based on intensity similarity.
    """
    # Create an output image to visualize the segmented region
    segmented_image = np.zeros(image.shape[:2], dtype=np.uint8)

    for seed in seeds:
        # Initialize the region growing queue with the seed point
        queue = [seed]
        seed_intensity = image[seed]

        while len(queue) > 0:
            current_pixel = queue.pop(0) # (row, col) format
            segmented_image[current_pixel] = 255  # Mark the current pixel as part of the region

            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:

                    if dy == 0 and dx == 0:
                        continue # Skip the current pixel

                    # Get the neighbor pixel coordinates
                    ny, nx = current_pixel[0] + dy, current_pixel[1] + dx # (row, col) format

                    # Ensure we are within image bounds
                    if ny >= 0 and ny < image.shape[0] and nx >= 0 and nx < image.shape[1]:

                        # If the pixel is not already part of the segmented region
                        if segmented_image[ny, nx] == 0:
                            neighbor_intensity = image[ny, nx]

                            # Check if the intensity difference is within the threshold
                            if np.linalg.norm(np.int32(neighbor_intensity) - np.int32(seed_intensity)) < threshold:
                                
                                queue.append((ny, nx)) # Add the neighbor to the queue for further exploration of his neighbors
                                segmented_image[ny, nx] = 255  # Mark as processed and part of the region

    return segmented_image

# Load the RGB image
image_path = 'images/test1.png'
img = cv2.imread(image_path)

# Resize the image to fit the screen (e.g., 50% of the original size)
scale_percent = 75
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize the image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
img_display = img.copy()

# Display the image to select seed points
img_display = img.copy()
cv2.imshow("Select Seeds", img_display)
cv2.setMouseCallback("Select Seeds", on_mouse_click)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()

# Convert the image to grayscale for easier intensity comparison
gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Perform region growing
segmentation_result = region_growing(gray_image, seeds, threshold)

# Overlay the segmentation result on the original image
overlay = img.copy()
overlay[segmentation_result == 255] = [255, 0, 0]  # Red color for segmented region

# Display the final result
cv2.imshow("Segmentation result", overlay)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
