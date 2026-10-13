import os

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def count_images(folder):
    if not os.path.exists(folder):
        return 0

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    return sum(
        1
        for file in os.listdir(folder)
        if file.lower().endswith(valid_extensions)
    )


print("=" * 50)
print("FER-2013 DATASET CHECK")
print("=" * 50)


# ==============================
# CHECK TRAINING DATA
# ==============================

print("\nTRAINING DATASET")
print("-" * 30)

total_train = 0

for emotion in EMOTIONS:

    path = os.path.join(TRAIN_PATH, emotion)

    count = count_images(path)

    total_train += count

    print(f"{emotion:10} : {count}")


print(f"\nTotal training images: {total_train}")


# ==============================
# CHECK TEST DATA
# ==============================

print("\nTEST DATASET")
print("-" * 30)

total_test = 0

for emotion in EMOTIONS:

    path = os.path.join(TEST_PATH, emotion)

    count = count_images(path)

    total_test += count

    print(f"{emotion:10} : {count}")


print(f"\nTotal test images: {total_test}")


# ==============================
# FINAL RESULT
# ==============================

print("\n" + "=" * 50)

if total_train > 0 and total_test > 0:

    print("Dataset loaded successfully!")

    print("Training images:", total_train)

    print("Testing images :", total_test)

    print("Total images   :", total_train + total_test)

else:

    print("Dataset structure is incorrect.")

    print("Please check dataset/train and dataset/test.")

print("=" * 50)