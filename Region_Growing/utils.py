import cv2

def display_image(window_name, img):
    """
    Display an image in a window with the given name.
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def resize_image(img, scale_percent = 65):
    """
    Resize the image by the given scale percentage.
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return img