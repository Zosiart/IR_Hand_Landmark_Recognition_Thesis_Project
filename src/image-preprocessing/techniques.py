import cv2
import numpy as np
from rembg import remove

class Techniques:

    def convert_to_greyscale(image):
        """Converts an image to greyscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def enhance_contrast_brightness(image, contrast=1.5, brightness=50):
        """Enhances the contrast and brightness of an image"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def detect_edges_canny(image, minVal=100, maxVal=200):
        """Detects edges in an image using the Canny edge detection algorithm"""
        return cv2.Canny(image, minVal, maxVal)

    def detect_edges_sobelX(image):
        """Detects edges in an image using the Sobel edge detection algorithm"""
        return cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)

    def detect_edges_sobelY(image):
        """Detects edges in an image using the Sobel edge detection algorithm"""
        return cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

    def histogram_equalization(image):
        """Applies histogram equalization to an image"""
        return cv2.equalizeHist(image)

    def clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image"""
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(image)

    def gaussian_blur(image, kernel_size=(3, 3), sigmaX=1.0):
        """Applies a Gaussian blur to an image"""
        return cv2.GaussianBlur(image, kernel_size, sigmaX)

    def median_blur(image, sigmaX=1.0):
        """Applies a median blur to an image"""
        return cv2.medianBlur(image, sigmaX)

    def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
        """Applies a bilateral filter to an image"""
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    def sharpen(image):
        """Sharpens an image using a kernel"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def invert_colors(image):
        """Inverts the colors of an image"""
        return cv2.bitwise_not(image)

    def inverse_transform(image):
        """Inverts the colors of an image"""
        return 255 - image

    def morphological_dilation(image):
        """Applies morphological dilation"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.dilate(image, kernel, iterations=1)

    def morphological_erosion(image):
        """Applies morphological erosion to an image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.erode(image, kernel, iterations=1)

    def morphological_opening(image):
        """Applies morphological opening to an image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def morphological_closing(image):
        """Applies morphological closing to an image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def laplacian(image):
        """Applies the Laplacian operator to an image"""
        return cv2.Laplacian(image, cv2.CV_64F)

    def remove_background(image):
        # Removing the background from the given Image
        output = remove(image)
        return output

    def invert_colors(image):
        """Inverts the colors of an image"""
        return cv2.bitwise_not(image)



