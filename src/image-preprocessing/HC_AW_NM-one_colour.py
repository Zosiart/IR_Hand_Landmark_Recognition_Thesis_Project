import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import canny

from techniques import Techniques

image_original = cv2.imread('../../resources/hand-pictures/frame15.jpg')
greyscale = Techniques.convert_to_greyscale(image_original)

cannyEdges = Techniques.detect_edges_canny(greyscale)
sobelXEdges = Techniques.detect_edges_sobelX(greyscale)
sobelYEdges = Techniques.detect_edges_sobelY(greyscale)
laplacian = Techniques.laplacian(image_original)

titles = ['Original', 'Greyscale', 'Canny Edges', 'Sobel X Edges', 'Sobel Y Edges', 'Laplacian']
images = [image_original, greyscale, cannyEdges, sobelXEdges, sobelYEdges, laplacian]

# Plot the results for comparison
plt.figure(figsize=(15, 10))

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()


############# Continuation of Sobel Y Edges ################

# Normalize and convert Sobel Y edges to uint8
sobelYEdges_uint8 = cv2.convertScaleAbs(sobelYEdges)

smoothed = cv2.GaussianBlur(sobelYEdges_uint8, (5, 5), 0)
smoothedMedian = Techniques.median_blur(sobelYEdges_uint8, 5)
bilateral = Techniques.bilateral_filter(sobelYEdges_uint8)

# Apply Canny edge detection on the normalized Sobel Y edges
cannyEdges2 = cv2.Canny(smoothed, 100, 200)

titles = ['Smoothed Gaussian', 'Smoothed Median', 'Bilateral Filter','Canny edges on Sobel Y Edges']
images = [smoothed, smoothedMedian, bilateral, cannyEdges2]

# Plot the results for comparison
plt.figure(figsize=(15, 10))

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Normalize and convert `commonEdges` to uint8 format
smoothed_uint_8 = cv2.convertScaleAbs(smoothed)

# Save the processed image to the specified path
output_path = '../../resources/stylized-pictures/HC_AW_NM-one_colour/smoothedGaussian.jpg'
cv2.imwrite(output_path, smoothed_uint_8)

print(f"Saved 'smoothedGaussian.jpg' at: {output_path}")


############## Combining Sobel X and Sobel Y Edges ################

# Normalize and convert Sobel edges to uint8
sobelXEdges_uint8 = cv2.convertScaleAbs(sobelXEdges)
sobelYEdges_uint8 = cv2.convertScaleAbs(sobelYEdges)

# Find common edges using bitwise AND
commonEdges = cv2.bitwise_and(sobelXEdges_uint8, sobelYEdges_uint8)

titles = ['Common Edges']
images = [commonEdges]

# Plot the results for comparison
plt.figure(figsize=(15, 10))

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()


# Normalize and convert `commonEdges` to uint8 format
commonEdges_uint8 = cv2.convertScaleAbs(commonEdges)

# Save the processed image to the specified path
output_path = '../../resources/stylized-pictures/HC_AW_NM-one_colour/common_edges.jpg'
cv2.imwrite(output_path, commonEdges_uint8)

print(f"Saved 'common_edges.jpg' at: {output_path}")


################### Creating a binary mask ####################

# Step 4: Threshold the combined edges to create a binary mask
_, binaryMask = cv2.threshold(commonEdges, 50, 255, cv2.THRESH_BINARY)

# Step 5: Convert greyscale image to 3-channel (BGR) to match the binary mask
greyscale3Channel = cv2.cvtColor(greyscale, cv2.COLOR_GRAY2BGR)

# Step 6: Use the mask to extract the hand area from the original image
# Mask needs to be 3-channel (BGR) to work with the original image
binaryMask3Channel = cv2.cvtColor(binaryMask, cv2.COLOR_GRAY2BGR)

# Apply the mask to the original image
handRegion = cv2.bitwise_and(greyscale3Channel, binaryMask3Channel)

# Step 7: Optionally, refine the mask with morphological operations (like dilation)
kernel = np.ones((5, 5), np.uint8)
dilatedMask = cv2.dilate(binaryMask, kernel, iterations=1)

opening = Techniques.morphological_opening(binaryMask)
opening = Techniques.morphological_opening(opening)

# Refine the hand region using the dilated mask
refinedHandRegion = cv2.bitwise_and(image_original, cv2.cvtColor(dilatedMask, cv2.COLOR_GRAY2BGR))

# Step 8: Show the results

# Plot the results
plt.figure(figsize=(10, 7))
plt.subplot(2, 3, 1)
plt.imshow(greyscale, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(2, 3, 2)
plt.imshow(sobelXEdges_uint8, cmap='gray')
plt.title('Sobel X')

plt.subplot(2, 3, 3)
plt.imshow(sobelYEdges_uint8, cmap='gray')
plt.title('Sobel Y')

plt.subplot(2, 3, 4)
plt.imshow(commonEdges, cmap='gray')
plt.title('Combined Sobel Edges')

plt.subplot(2, 3, 5)
plt.imshow(binaryMask, cmap='gray')
plt.title('Binary Mask')

plt.subplot(2, 3, 6)
plt.imshow(refinedHandRegion)
plt.title('Extracted Hand Region')

plt.tight_layout()
plt.show()

extracted_uint8 = cv2.convertScaleAbs(refinedHandRegion)
output_path = '../../resources/stylized-pictures/HC_AW_NM-one_colour/extracted_dilated_mask.jpg'
cv2.imwrite(output_path, extracted_uint8)


opened = Techniques.morphological_opening(refinedHandRegion)
closed = Techniques.morphological_closing(refinedHandRegion)

denoised = cv2.fastNlMeansDenoising(refinedHandRegion)


# show plot of opened and closed and denoised
titles = ['Binary Mask', 'Dilated', 'Binary Mask (After Opening)']
images = [binaryMask, dilatedMask, opening]

# Plot the results for comparison
plt.figure(figsize=(15, 10))

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# plot original sobelEdgesX, sobelEdgesY, opening
titles = ['Original', 'Sobel X Edges', 'Sobel Y Edges', 'Binary Mask (After Opening)']
images = [image_original, sobelXEdges, sobelYEdges, opening]

# Plot the results for comparison
plt.figure(figsize=(15, 10))

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

