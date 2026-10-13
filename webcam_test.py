import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/emotion_model.keras"

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

IMG_SIZE = 48


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")


# ============================================================
# LOAD OPENCV FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade.")
    exit()

print("OpenCV face detector loaded successfully!")


# ============================================================
# START WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started!")
print("Press Q to quit.")


# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    # Read frame from webcam
    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    # Flip image horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ========================================================
    # PROCESS EACH FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y + h, x:x + w]

        # Resize to CNN input size
        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        # Convert to float32
        #
        # IMPORTANT:
        # The trained model already contains Rescaling(1/255),
        # so DO NOT divide by 255 here.
        face = face.astype("float32")

        # Add batch dimension and channel dimension
        #
        # Before:
        # (48, 48)
        #
        # After:
        # (1, 48, 48, 1)
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # ====================================================
        # CNN PREDICTION
        # ====================================================

        predictions = model.predict(
            face,
            verbose=0
        )[0]

        # Get highest probability
        predicted_index = np.argmax(predictions)

        predicted_emotion = CLASS_NAMES[predicted_index]

        confidence = predictions[predicted_index] * 100

        # ====================================================
        # DRAW FACE RECTANGLE
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

        # ====================================================
        # DISPLAY EMOTION
        # ====================================================

        label = f"{predicted_emotion} {confidence:.1f}%"

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # ========================================================
    # DISPLAY FACE COUNT
    # ========================================================

    face_count = len(faces)

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show webcam window
    cv2.imshow(
        "EMO VISION - Facial Expression Recognition",
        frame
    )

    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("Webcam test completed.")