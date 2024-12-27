from PIL import Image, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the provided images
path_original = "../../resources/hand-pictures/frame 4 - no mould.jpg"

image_original = Image.open(path_original)

# Convert to grayscale for processing
original_gray = cv2.cvtColor(np.array(image_original), cv2.COLOR_BGR2GRAY)

# 1. Enhance contrast and brightness
def enhance_contrast_brightness(image, contrast=1.5, brightness=50):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

enhanced_image = enhance_contrast_brightness(original_gray)

# 2. Apply edge detection
edges = cv2.Canny(original_gray, 50, 150)

# 3. Histogram equalization
hist_eq = cv2.equalizeHist(original_gray)

# 4. Selective smoothing to reduce noise
smoothed = cv2.GaussianBlur(original_gray, (9, 9), 0)

# Plot the results for comparison
plt.figure(figsize=(15, 10))

titles = ['Original', 'Enhanced Contrast/Brightness', 'Edge Detection', 'Histogram Equalization', 'Smoothed']
images = [original_gray, enhanced_image, edges, hist_eq, smoothed]

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

########### More distinctive ################

# 1. Apply thresholding after histogram equalization
_, thresholded = cv2.threshold(hist_eq, 100, 255, cv2.THRESH_BINARY)

# 2. Sharpening using a kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(enhanced_image, -1, sharpen_kernel)

# 3. Invert colors (negative transformation)
inverted = cv2.bitwise_not(original_gray)

# 4. Combine edge detection with histogram equalization
edges_on_hist_eq = cv2.Canny(hist_eq, 50, 150)

# 5. Combine smoothing and thresholding
smoothed_threshold = cv2.GaussianBlur(thresholded, (9, 9), 0)

# Plot the advanced transformations
plt.figure(figsize=(15, 15))

titles = [
    'Thresholding on Histogram Equalization',
    'Sharpened Image',
    'Color Inversion',
    'Edges on Histogram Equalized Image',
    'Smoothed + Thresholding'
]
images = [thresholded, sharpened, inverted, edges_on_hist_eq, smoothed_threshold]

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(3, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

############ Sharpened Image continuation ##########

# 1. Apply adaptive thresholding to the sharpened image
adaptive_thresh = cv2.adaptiveThreshold(
    sharpened,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# 2. Selective smoothing on the sharpened image to reduce noise
selective_smoothing = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

# 3. Combine edge detection with the sharpened image
edges_on_sharpened = cv2.Canny(sharpened, 50, 150)

# 4. Highlight contours based on thresholding results
contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoured_image = sharpened.copy()
cv2.drawContours(contoured_image, contours, -1, (255, 255, 255), thickness=1)

# Plot the results of further refinement
plt.figure(figsize=(15, 10))

titles = [
    'Adaptive Thresholding on Sharpened',
    'Selective Smoothing on Sharpened',
    'Edges on Sharpened',
    'Contours Highlighted'
]
images = [adaptive_thresh, selective_smoothing, edges_on_sharpened, contoured_image]

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()


############# My try ####################

edges = cv2.Canny(contoured_image, 50, 150)

# Plot the results of further refinement
plt.figure(figsize=(20,20))

titles = [
    'Sharpened on Contours',
    'Edges'
]
images = [sharpened, edges]

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

##### My try - smoothed ############### frame 4
edges = cv2.Canny(smoothed, 50, 150)

# Plot the results of further refinement
plt.figure(figsize=(20,20))

titles = [
    'Smoothed',
    'Edges'
]
images = [smoothed, edges]

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

