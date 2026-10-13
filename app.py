from flask import Flask, render_template, request, jsonify,render_template, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import base64
import os


app = Flask(__name__)
@app.route("/test")
def test():
    return "TEST ROUTE WORKING"


MODEL_PATH = "model/emotion_model.keras"

IMG_SIZE = (48, 48)

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# LOAD CNN MODEL
# ============================================================

print("=" * 60)
print("EMO VISION")
print("Real-Time Facial Expression Recognition")
print("=" * 60)

print()
print("Loading CNN model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded successfully.")

except Exception as e:
    print("ERROR: Could not load CNN model.")
    print("Reason:", e)
    model = None


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print()
print("Loading OpenCV face detector...")

try:

    CASCADE_PATH = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)

    if face_detector.empty():
        print("ERROR: Haar Cascade could not be loaded.")
        face_detector = None

    else:
        print("OpenCV face detector loaded successfully.")

except Exception as e:

    print("ERROR: Could not initialize face detector.")
    print("Reason:", e)
    face_detector = None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# DECODE BASE64 IMAGE
# ============================================================

def decode_image(image_data):

    try:

        # Remove data:image/jpeg;base64,... part
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode Base64
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to NumPy array
        np_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Convert NumPy array to OpenCV image
        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        return frame

    except Exception as e:

        print("Image decoding error:", e)

        return None


# ============================================================
# PREPROCESS FACE
# ============================================================

def preprocess_face(face):

    """
    Converts detected face into the format expected
    by the trained CNN model.

    Model input:
        48 x 48
        Grayscale
        float32

    IMPORTANT:
    The trained CNN already contains a Rescaling(1/255)
    layer, so we DO NOT divide the image by 255 here.
    """

    # Convert BGR to grayscale
    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to CNN input size
    gray = cv2.resize(
        gray,
        IMG_SIZE
    )

    # Convert to float32
    gray = gray.astype("float32")

    # Add channel dimension
    # 48 x 48
    # becomes 48 x 48 x 1
    gray = np.expand_dims(
        gray,
        axis=-1
    )

    # Add batch dimension
    # becomes 1 x 48 x 48 x 1
    gray = np.expand_dims(
        gray,
        axis=0
    )

    return gray


# ============================================================
# PREDICT EXPRESSION
# ============================================================

def predict_expression(face):

    # Preprocess detected face
    processed_face = preprocess_face(face)

    # CNN prediction
    predictions = model.predict(
        processed_face,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = int(
        np.argmax(predictions)
    )

    # Get expression name
    expression = CLASS_NAMES[predicted_index]

    # Convert confidence to percentage
    confidence = float(
        predictions[predicted_index] * 100
    )

    # Create probability dictionary
    probabilities = {}

    for index, class_name in enumerate(CLASS_NAMES):

        probabilities[class_name] = round(
            float(predictions[index] * 100),
            2
        )

    return (
        expression,
        confidence,
        probabilities
    )


# ============================================================
# PREDICT ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "success": False,
                "error": "CNN model is not loaded."
            }), 500


        # ----------------------------------------------------
        # Check face detector
        # ----------------------------------------------------

        if face_detector is None:

            return jsonify({
                "success": False,
                "error": "Face detector is not available."
            }), 500


        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No image data received."
            }), 400


        # ----------------------------------------------------
        # Get image
        # ----------------------------------------------------

        image_data = data.get("image")

        if not image_data:

            return jsonify({
                "success": False,
                "error": "Image data is missing."
            }), 400


        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------

        frame = decode_image(image_data)

        if frame is None:

            return jsonify({
                "success": False,
                "error": "Unable to decode image."
            }), 400


        # ----------------------------------------------------
        # Convert image to grayscale
        # ----------------------------------------------------

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results = []


        # ----------------------------------------------------
        # Process every detected face
        # ----------------------------------------------------

        for (x, y, width, height) in faces:

            # Extract face region
            face = frame[
                y:y + height,
                x:x + width
            ]


            # Skip invalid face
            if face.size == 0:
                continue


            # Predict expression
            (
                expression,
                confidence,
                probabilities
            ) = predict_expression(face)


            # Store result
            results.append({

                "x": int(x),

                "y": int(y),

                "width": int(width),

                "height": int(height),

                "expression": expression,

                "confidence": round(
                    confidence,
                    2
                ),

                "probabilities": probabilities
            })


        # ----------------------------------------------------
        # Return prediction result
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "face_count": len(results),

            "faces": results

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print("Prediction error:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "model_loaded": model is not None,

        "face_detector_loaded": face_detector is not None,

        "classes": CLASS_NAMES,

        "input_size": "48x48",

        "project": "EMO VISION"

    })
@app.route("/model-graph/<filename>")
def model_graph(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
        filename
    )
    print("Graph folder:", graph_folder)
    print("Requested graph:", filename)

    return send_from_directory(graph_folder, filename)
# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("EMO VISION")
    print("Real-Time Facial Expression Recognition")
    print("=" * 60)
    print()

    print("CNN Model:",
          "READY" if model is not None else "NOT READY")

    print(
        "OpenCV Face Detector:",
        "READY" if face_detector is not None else "NOT READY"
    )

    print("Flask: READY")

    print()
    print("Supported Expressions:")

    for expression in CLASS_NAMES:
        print(" -", expression)

    print()
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )