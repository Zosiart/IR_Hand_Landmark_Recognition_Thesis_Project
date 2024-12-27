import cv2
from matplotlib import pyplot as plt

scream_path = '../../resources/style-pictures/scream.jpg'
hands_path = '../../resources/hand-pictures/frame1.jpg'
stylized_picture = '../../resources/stylized-pictures/hands-scream.jpg'

# Load the images
scream = cv2.imread(scream_path)
hands = cv2.imread(hands_path)
result = cv2.imread(stylized_picture)

# Plot the images
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(scream, cv2.COLOR_BGR2RGB))
plt.title('Scream')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(hands, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Stylized Picture')
plt.axis('off')

plt.tight_layout()
plt.show()