import cv2
import matplotlib.pyplot as plt
import os

image_path = '../../resources/evaluation_dataset/HC_AW_NM/frame6.jpg'

# Verify if the file exists
if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
else:
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load the image. Check if the file is corrupted or the format is unsupported.")
    else:
        # Invert the image
        inverted_image = cv2.bitwise_not(image)

        # Display the images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Inverted Image')
        plt.imshow(inverted_image, cmap='gray')

        plt.tight_layout()
        plt.show()
