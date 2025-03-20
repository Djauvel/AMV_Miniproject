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
            image_path = os.path.join(save_dir, f"image_{i+1}.jpg")
            cv.imwrite(image_path, frame)
            print(f"Image {i+1} saved at {image_path}")
            time.sleep(2)
        else:
            print(f"Error: Could not capture image {i+1}")
            time.sleep(2)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images from a USB webcam.")
    parser.add_argument("num_images", type=int, help="Number of images to capture")
    parser.add_argument("--save_dir", type=str, default="captured_images", help="Directory to save the images")
    
    args = parser.parse_args()
    
    capture_images(args.num_images, args.save_dir)