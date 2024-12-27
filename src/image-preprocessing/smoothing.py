import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
image_path = "../../resources/hand-pictures/frame2.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Convert to grayscale for methods that require it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Bilateral Filtering
bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# 2. CLAHE + Bilateral Filtering
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_gray = clahe.apply(gray)
clahe_bilateral = cv2.bilateralFilter(clahe_gray, d=9, sigmaColor=50, sigmaSpace=50)

# 3. Edge-Aware Smoothing
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
blurred = cv2.GaussianBlur(image, (7, 7), 1)
edge_aware = cv2.bitwise_and(blurred, blurred, mask=edges)

# 4. Non-Local Means Denoising
nlm_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 5. Guided Filtering (requires ximgproc module in OpenCV)
try:
    guided = cv2.ximgproc.guidedFilter(guide=gray, src=gray, radius=8, eps=0.1**2)
    guided_filtered = cv2.merge([guided] * 3)  # Convert back to 3 channels
except AttributeError:
    guided_filtered = np.zeros_like(image)  # Placeholder if guided filter isn't available

# Plotting results
methods = {
    "Original": image,
    "Bilateral Filtering": bilateral,
    "CLAHE + Bilateral": clahe_bilateral,
    "Edge-Aware Smoothing": edge_aware,
    "Non-Local Means Denoising": nlm_denoised,
    "Guided Filtering": guided_filtered,
}

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, result) in zip(axs.flatten(), methods.items()):
    if len(result.shape) == 2:  # Grayscale images
        ax.imshow(result, cmap="gray")
    else:  # Color images
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()


################## Trying Canny #####################
edgesOriginal = cv2.Canny(image, 50, 150)
edgesBilateral = cv2.Canny(bilateral, 50, 150)
edgesCLAHEBilateral = cv2.Canny(clahe_bilateral, 50, 150)
edgesNlm = cv2.Canny(nlm_denoised, 50, 150)
edgesGuided = cv2.Canny(guided_filtered, 50, 150)


methods = {
    "Original": edgesOriginal,
    "Bilateral Filtering": edgesBilateral,
    "CLAHE + Bilateral": edgesCLAHEBilateral,
    "Non-Local Means Denoising": edgesNlm,
    "Guided Filtering": edgesGuided,
}

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, result) in zip(axs.flatten(), methods.items()):
    if len(result.shape) == 2:  # Grayscale images
        ax.imshow(result, cmap="gray")
    else:  # Color images
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()