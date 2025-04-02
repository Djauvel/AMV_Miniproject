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

image = cv2.imread('dataset/MachineScrew/image_193.jpg')


im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(im, (5, 5), 0)


# Perform Canny edge detection
edges = cv2.Canny(blurred, 100, 200)


# Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
contoured_image = image.copy()
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)

# Display the contoured image
plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.show()

# Get the bounding box of the largest contour (assumed to be the screw)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
print(f"Length (height): {h}, Width: {w}")















# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 500
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
 detector = cv2.SimpleBlobDetector(params)
else : 
 detector = cv2.SimpleBlobDetector_create(params)



# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)

#cv2.imshow("Greyscale Image", im)
#v2.waitKey(0)
#cv2.destroyAllWindows()
