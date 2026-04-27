import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ==============================
# STEP 1: Download dataset
# ==============================
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
print("Dataset Path:", path)

# ==============================
# STEP 2: Find correct folder
# ==============================
# Usually dataset contains a folder like "data"
DATASET_PATH = os.path.join(path, "data")  # 👈 adjust if needed

print("Using dataset folder:", DATASET_PATH)

# ==============================
# STEP 3: Image settings
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ==============================
# STEP 4: Model
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# STEP 5: Train
# ==============================
model.fit(train_data, validation_data=val_data, epochs=5)

model.save("mask_model.h5")

print("✅ Training complete!")