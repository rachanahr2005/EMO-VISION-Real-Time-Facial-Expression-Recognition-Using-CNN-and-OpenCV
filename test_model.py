import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/emotion_model.keras"
TEST_DIR = "dataset/test"

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
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("-" * 60)


# ============================================================
# TEST ONE RANDOM IMAGE FROM EACH CLASS
# ============================================================

print("\nTesting one random image from each expression class:\n")

correct = 0
total = 0


for class_name in CLASS_NAMES:

    class_dir = os.path.join(TEST_DIR, class_name)

    # Get image files
    images = [
        file_name
        for file_name in os.listdir(class_dir)
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    if not images:
        print(f"No images found for: {class_name}")
        continue

    # Select random image
    file_name = random.choice(images)

    image_path = os.path.join(class_dir, file_name)

    # Load image as grayscale and resize to 48x48
    image = load_img(
        image_path,
        color_mode="grayscale",
        target_size=IMG_SIZE
    )

    # Convert image to NumPy array
    image_array = img_to_array(image).astype("float32")

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index] * 100

    # Check whether prediction is correct
    is_correct = predicted_class == class_name

    if is_correct:
        correct += 1

    total += 1

    result = "✓ CORRECT" if is_correct else "✗ WRONG"

    print(
        f"Actual: {class_name:<8} | "
        f"Predicted: {predicted_class:<8} | "
        f"Confidence: {confidence:6.2f}% | "
        f"{result}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

if total > 0:
    accuracy = (correct / total) * 100

    print(f"Correct predictions : {correct}/{total}")
    print(f"Random sample accuracy: {accuracy:.2f}%")

print("=" * 60)

print("\nModel testing completed!")