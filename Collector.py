import cv2 as cv
import os
import argparse
import time

def capture_images(num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    for i in range(num_images):
        ret, frame = cap.read()
        if ret:
            image_path = os.path.join(save_dir, f"image_{i+16}.jpg")
            cv.imwrite(image_path, frame)
            print(f"Image {i+1} saved at {image_path}")
            time.sleep(2)
        else:
            print(f"Error: Could not capture image {i+1}")
            time.sleep(2)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    
    num_images = 15
    save_dir = "captured_images"
    print(f"Capturing {num_images} images and saving to {save_dir}...")
    capture_images(num_images, save_dir)