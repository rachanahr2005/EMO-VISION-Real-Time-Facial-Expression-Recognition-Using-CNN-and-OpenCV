import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=SEED,
    shuffle=True
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    shuffle=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# IMPROVE DATA PIPELINE SPEED
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

for images, labels in train_ds:
    labels = labels.numpy()

    for label in labels:
        class_counts[label] += 1

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=np.repeat(
        np.arange(len(CLASS_NAMES)),
        class_counts
    )
)

class_weights = {
    i: float(class_weights_array[i])
    for i in range(len(CLASS_NAMES))
}

print("\nClass weights:")

for i, name in enumerate(CLASS_NAMES):
    print(f"{name}: {class_weights[i]:.4f}")


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")


# ============================================================
# BUILD LIGHTWEIGHT CNN
# ============================================================

print("\nBuilding CNN model...")

model = models.Sequential([

    layers.Input(shape=(48, 48, 1)),

    # Normalize pixels
    layers.Rescaling(1.0 / 255),

    # Data augmentation
    data_augmentation,

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.20),

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),

    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.30),

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.40),

    layers.Dense(
        7,
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    "model/emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n===================================================")
print("STARTING CNN TRAINING")
print("===================================================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    "model/emotion_model.keras"
)


# ============================================================
# TEST MODEL
# ============================================================

print("\n===================================================")
print("EVALUATING MODEL")
print("===================================================\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n===================================================")
print("CLASSIFICATION REPORT")
print("===================================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest")

plt.title("Emotion Recognition Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

# Display numbers inside the matrix
threshold = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=200
)

plt.close()

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    "model/emotion_model.keras"
)

print("\n===================================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================================")

print("\nFiles created inside model folder:")

print("1. emotion_model.keras")
print("2. accuracy_graph.png")
print("3. loss_graph.png")
print("4. confusion_matrix.png")