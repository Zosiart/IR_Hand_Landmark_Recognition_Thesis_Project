import cv2
import numpy as np

from src.colorization.Zhang import create_colorized_pictures

def colorize_zhang_eccv16(image_path):
    create_colorized_pictures(model='eccv16', img_path=image_path)
    return './resources/stylized_pictures/eccv16/saved_eccv16.png'

def colorize_zhang_siggraph17(image_path, image_name):
    create_colorized_pictures(model='siggraph17', img_path=image_path, save_prefix=image_name)
    return f'../../resources/stylized-pictures/siggraph17/{image_name}_siggraph17.png'


def clahe(image, clipLimit=2.0, tileGridSize=(8, 8), save_path=None, save_as_rgb=True):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Parameters:
    - image: Input image (grayscale or BGR/RGB).
    - clipLimit: Threshold for contrast limiting.
    - tileGridSize: Size of the grid for histogram equalization.
    - save_path: Path to save the output image.
    - save_as_rgb: Save the CLAHE-applied image in RGB format if True.

    Returns:
    - Processed image in grayscale or RGB format.
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:  # If RGB/BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_image = clahe.apply(gray)

    # Convert back to RGB if required
    if save_as_rgb:
        clahe_image_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
    else:
        clahe_image_rgb = clahe_image

    # Save the image if a save_path is provided
    if save_path:
        cv2.imwrite(save_path, clahe_image_rgb if save_as_rgb else clahe_image)
        print(f"CLAHE-applied image saved at {save_path}")

    return clahe_image_rgb if save_as_rgb else clahe_image


def reduce_brightness(image_path, save_path, reduction_factor=0.5):
    """
    Reads an image, reduces its brightness, and saves the result.

    Parameters:
    - image_path: Path to the input image.
    - save_path: Path to save the output image.
    - reduction_factor: Factor by which to reduce brightness (0.0 to 1.0, where 1.0 is no change).

    Returns:
    - The brightness-reduced image as a NumPy array.
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Convert to float to avoid clipping during multiplication
    image = image.astype(np.float32)

    # Reduce brightness
    brightness_reduced = np.clip(image * reduction_factor, 0, 255).astype(np.uint8)

    # Save the result
    cv2.imwrite(save_path, brightness_reduced)
    print(f"Brightness-reduced image saved at {save_path}")

    return brightness_reduced


