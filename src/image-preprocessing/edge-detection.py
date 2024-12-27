import cv2
import os

def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=1.0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 70, 135)

    return blurred, edges

# Paths
input_path = '../../resources/hand-pictures/frame2.jpg'
output_folder = '../../resources/hand-pictures/processed/'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the image
img = cv2.imread(input_path)

if img is None:
    print("Error: Image not found!")
else:
    # Perform edge detection
    blurred, edges = canny_edge_detection(img)

    # Save the processed images
    blurred_path = os.path.join(output_folder, 'blurred.jpg')
    edges_path = os.path.join(output_folder, 'edges.jpg')

    cv2.imwrite(blurred_path, blurred)
    cv2.imwrite(edges_path, edges)

    print(f"Blurred image saved to: {blurred_path}")
    print(f"Edges image saved to: {edges_path}")

    # Define a kernel for morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply dilation (expands edges)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    dilated_path = os.path.join(output_folder, 'dilated.jpg')
    cv2.imwrite(dilated_path, dilated)
    print(f"Dilated image saved to: {dilated_path}")

    # # Apply erosion (shrinks edges)
    # eroded = cv2.erode(edges, kernel, iterations=1)
    # eroded_path = os.path.join(output_folder, 'eroded.jpg')
    # cv2.imwrite(eroded_path, eroded)
    # print(f"Eroded image saved to: {eroded_path}")

    # Apply closing (dilation followed by erosion - closes small holes)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    closed_path = os.path.join(output_folder, 'closed.jpg')
    cv2.imwrite(closed_path, closed)
    print(f"Closed image saved to: {closed_path}")

    # Apply opening (erosion followed by dilation - removes noise)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    opened_path = os.path.join(output_folder, 'opened.jpg')
    cv2.imwrite(opened_path, opened)
    print(f"Opened image saved to: {opened_path}")



    # # Apply gradient (difference between dilation and erosion)
    # gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    # gradient_path = os.path.join(output_folder, 'gradient.jpg')
    # cv2.imwrite(gradient_path, gradient)
    # print(f"Gradient image saved to: {gradient_path}")
