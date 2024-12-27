import cv2

# read the input image
img = cv2.imread('../../resources/hand-pictures/frame1.jpg')

# Define the contrast and brightness values
contrast = 1.0  # Try values between 3 and 10 for strong contrast
brightness = -90  # Try values between 5 and 50 for moderate brightness

# Adjust contrast and brightness using addWeighted
out = cv2.addWeighted(img, contrast, img, 0, brightness)

cv2.imwrite('../../resources/stylized-pictures/adjusted_brightness.jpg', out)
# Display the image with changed contrast and brightness
# cv2.imshow('Adjusted', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
