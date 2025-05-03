import cv2
import os

# Folder to save images
save_dir = "screw_capture"
os.makedirs(save_dir, exist_ok=True)

# Automatically count existing images
existing_images = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
img_counter = len(existing_images)

# Open webcam
cam = cv2.VideoCapture(0)
cv2.namedWindow("Screw Collector")

print("Press 'c' to capture an image. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Screw Collector", frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        img_name = os.path.join(save_dir, f"screw_{img_counter:04d}.jpg")
        cv2.imwrite(img_name, frame)
        img_counter += 1
        print(f"Captured {img_name} — Total images: {img_counter}")

    elif key == ord('q'):
        print("Exiting.")
        break

cam.release()
cv2.destroyAllWindows()