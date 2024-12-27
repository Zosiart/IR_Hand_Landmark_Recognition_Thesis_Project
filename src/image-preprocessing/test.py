import cv2
import numpy as np

# Load the infrared image
image = cv2.imread('../../resources/hand-pictures/frame2.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: Apply CLAHE to enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image)

# # Step 2: Apply adaptive thresholding to create a mask for the hand
# _, hand_mask = cv2.threshold(image_clahe, 200, 255, cv2.THRESH_BINARY)
#
# # Step 3: Darken the background by applying the mask to the original image
# background_suppressed = cv2.bitwise_and(image, image, mask=hand_mask)
# background_suppressed = cv2.addWeighted(background_suppressed, 1.5, background_suppressed, 0, -50)

# Step 4: Increase brightness and contrast selectively
# Adjust parameters for brightness and contrast to make the hand stand out more.
alpha = -1.0  # Contrast control (1.0-3.0)
beta = -50.   # Brightness control (0-100)
hand_enhanced = cv2.convertScaleAbs(image_clahe, alpha=alpha, beta=beta)

# Step 5: Apply morphological closing to fill small holes in the hand region
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
hand_enhanced = cv2.morphologyEx(hand_enhanced, cv2.MORPH_CLOSE, kernel)

# Save the final result as an image file
output_path = '../../resources/stylized-pictures/enhanced_hand_image.png'
cv2.imwrite(output_path, hand_enhanced)

print(f"Enhanced image saved at {output_path}")
