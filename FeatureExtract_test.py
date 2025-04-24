import os
import cv2
from pathlib import Path
import glob
import numpy as np

import matplotlib.pyplot as plt

# Path to the dataset folder
DataPath = Path('dataset/MachineScrew')

# Count images with specific extensions
image_count = len(list(DataPath.glob("*.jpg"))) + len(list(DataPath.glob("*.png"))) + len(list(DataPath.glob("*.jpeg")))

print(f"Number of images: {image_count}")

#image = cv2.imread('dataset/MachineScrew/image_193.jpg')
image = cv2.imread('examroomdataset/FrenchScrew/imag_1.jpg')

#im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#load the image in grayscale
#im = cv2.imread('dataset/MachineScrew/image_193.jpg', cv2.IMREAD_GRAYSCALE)

im = cv2.imread('examroomdataset/FrenchScrew/imag_1.jpg', cv2.IMREAD_GRAYSCALE)


blurred = cv2.GaussianBlur(im, (5, 5), 0)

## different methods of edge detection


# Perform Canny edge detection
edges = cv2.Canny(blurred, 100, 200)


# Apply binary thresholding
#_, thresh = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY_INV)


# Apply Adaptive Thresholding
#thresh = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                               cv2.THRESH_BINARY_INV, 135, 8)





#show the thresholded image
plt.imshow(edges, cmap='gray')
plt.title('Thresholded Image')
plt.show()


# Find contours in the image
#contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Draw the contours on the original image
contoured_image = image.copy()
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)

# Display the contoured image
plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.show()




# Get the largest contour (which should be the screw)
largest_contour = max(contours, key=cv2.contourArea)

# Get the rotated bounding box (angle of rotation)
rect = cv2.minAreaRect(largest_contour)
angle = rect[2]

#dimensions of the bounding box
width, height = rect[1]
print(f"Width: {width}, Height: {height}")
#aspect ratio of the bounding box as the longest side divided by the shortest side
aspect_ratio = max(width, height) / min(width, height)
print(f"Aspect Ratio: {aspect_ratio}")

#show the bounding box on the image
box = cv2.boxPoints(rect)
box = np.int0(box)

bounding_box_image = image.copy()
cv2.drawContours(bounding_box_image, [box], 0, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Bounding Box')
plt.show()
