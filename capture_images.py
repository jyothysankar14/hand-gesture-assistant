import cv2
import os

gesture_name = input("Enter the gesture name (must match folder name): ")
save_path = f"dataset/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame so it's mirror-like
    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture Gesture", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        count += 1
        img_name = os.path.join(save_path, f"{gesture_name}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
