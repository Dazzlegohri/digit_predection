#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:25:40 2026

@author: Dazzle
"""


import os
from glob import glob
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

images_path = "/Users/dazzle/Downloads/mnist-images/0-9 test images/0-9 test images"

images = sorted(
    glob(os.path.join(images_path, "*.png")) +
    glob(os.path.join(images_path, "*.jpg"))
)

print("Total training images:", len(images))

image_matrices = []
labels = []

for image in images:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (28, 28))
    image_matrices.append(img)

    label = int(os.path.basename(image)[0])
    labels.append(label)

# convert in numpy array

X = np.array(image_matrices, dtype=np.float32) / 255.0
y = np.array(labels, dtype=np.int32)

X = X.reshape(X.shape[0], -1)  

print("X shape:", X.shape)
print("y shape:", y.shape)



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


model.fit(
    X,
    y,
    epochs=30,
    batch_size=1,    
    shuffle=True
)


model.save("digit_model.h5")
print("✅ Model saved as digit_model.h5")
