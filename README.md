EMO-VISION: Real-Time Facial Expression Recognition

About the Project

EMO-VISION is a facial emotion recognition project that uses Computer Vision and Deep Learning to identify human facial expressions from images or webcam input.

Features

- Face detection using OpenCV
- Image preprocessing for facial images
- CNN-based emotion classification
- Flask web application
- Prediction confidence display
- Image upload and webcam-based testing

Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy
- HTML, CSS, JavaScript

Recognized Emotions

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Project Structure

- "app.py" — Flask application
- "train.py" — Model training
- "test_model.py" — Model testing
- "webcam_test.py" — Webcam testing
- "check_dataset.py" — Dataset checking
- "model/" — Trained model files
- "templates/" — HTML templates
- "static/" — CSS and JavaScript files
- "requirements.txt" — Python dependencies

Installation and Execution

1. Clone this repository.

2. Open the project folder in your terminal.

3. Create and activate a Python virtual environment.

4. Install dependencies:
   
   pip install -r requirements.txt

5. Run the Flask application:
   
   python app.py

6. Open the local URL displayed in the terminal in your browser.

Disclaimer

This project is intended for educational and demonstration purposes. Emotion predictions may be inaccurate and should not be treated as definitive assessments of a person's emotional state.

Author

Developed as a Deep Learning and Computer Vision project.
