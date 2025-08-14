# Hand Gesture Controlled Assistant

A deep learning–based assistant that recognizes hand gestures and responds to them—transforming gestures into actions with Python and OpenCV.

---

##  Features
- **Capture & Label** hand gesture images for dataset creation.
- **Train & Save** models for gesture classification.
- **Real-time Detection** using a webcam to recognize gestures.
- **Action Mapping**, enabling gestures to trigger custom assistant commands or responses.

---

##  Project Structure

hand-gesture-assistant/
├── dataset/ # Captured images of different gestures
├── capture_images.py # Script to capture and label gesture images
├── train_model.py # Train a machine learning model on the dataset
├── save_model.py # Save the trained model for deployment
├── app.py # Main application—uses webcam for live gesture recognition and actions
├── .gitignore # Files to ignore (like datasets, model weights)
└── README.md # Project documentation

---

##  Setup & Usage

1. Clone the repository:
```bash
git clone https://github.com/jyothysankar14/hand-gesture-assistant.git
cd hand-gesture-assistant

2. Install dependencies:
pip install -r requirements.txt

3. Capture Gesture Data:
python capture_images.py
- Use your webcam to capture labeled images for each gesture category.

4. Train the Model:
python train_model.py
- Trains a classifier (e.g., CNN) and evaluates its accuracy.

5. Save the Model:
python save_model.py
- Exports the trained model for inference.

6. Run the Gesture Assistant:
python app.py
- Launches the assistant—recognizes gestures live and executes mapped actions.

Notes & Tips

Dataset Quality: Ensure varied backgrounds and lighting when capturing gestures to improve robustness.

Model Configuration: Adjust hyperparameters (like epochs, batch size) in train_model.py for better performance.

Extensible Actions: In app.py, map gestures to system commands, spoken responses, or other Python functions.

License

This project is licensed under the MIT License—see the LICENSE file for details.

Acknowledgements

OpenCV for computer vision tools.

TensorFlow for model training.

The open source community for inspiring gesture-based interfaces.



