#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:03:33 2026

@author: dazzle
"""

import os
from glob import glob
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =========================
# ROOT TRAINING PATH
# =========================
root_path = "/Users/dazzle/Downloads/colorized-MNIST-master/training"

image_matrices = []
labels = []

# =========================
# LOOP THROUGH 0–9 FOLDERS
# =========================
for digit in range(10):
    folder_path = os.path.join(root_path, str(digit))

    if not os.path.isdir(folder_path):
        continue

    images = glob(os.path.join(folder_path, "*.png")) + \
             glob(os.path.join(folder_path, "*.jpg"))

    print(f"Loading digit {digit} → {len(images)} images")

    for image in images:
        img = cv2.imread(image)   # read color or grayscale

        if img is None:
            continue

        # color → grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))

        image_matrices.append(img)
        labels.append(digit)

# =========================
# NUMPY CONVERSION
# =========================
X = np.array(image_matrices, dtype=np.float32) / 255.0
y = np.array(labels, dtype=np.int32)

X = X.reshape(X.shape[0], -1)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# =========================
# MODEL
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(784,)),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    shuffle=True
)

# =========================
# SAVE MODEL
# =========================
model.save("digit_optimised_model.h5")
print("✅ Model saved as digit_optimised_model.h5")
