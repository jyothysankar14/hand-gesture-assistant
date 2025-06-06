import cv2
import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model('gesture_model.h5')

# Class labels (edit based on your folder names)
class_names = ['thumbs_up', 'thumbs_down', 'stop', 'peace']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict
    pred = model.predict(img)
    label = class_names[np.argmax(pred)]

    # Action
    if label == 'thumbs_up':
        pyautogui.press('volumeup')
    elif label == 'thumbs_down':
        pyautogui.press('volumedown')
    elif label == 'stop':
        pyautogui.press('space')  # Pause video
    elif label == 'peace':
        pyautogui.click()

    # Show webcam
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
