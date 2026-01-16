#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:24:04 2026

@author: dazzle
"""

import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
N = 1000                # Number of images to test
SHOW_IMAGES = True     # False if you don't want images

# =========================
# LOAD MODEL
# =========================
model = load_model("digit_optimised_model.h5")
print("✅ Model loaded")

# =========================
# TEST IMAGES PATH
# =========================
test_images_path = "/Users/dazzle/Downloads/mnist-images/0-9 test images/suffeled_data"

test_images = sorted(
    glob(os.path.join(test_images_path, "*.png")) +
    glob(os.path.join(test_images_path, "*.jpg"))
)

if len(test_images) == 0:
    raise ValueError("❌ No test images found. Check folder path.")

print("Total test images available:", len(test_images))
print("Images tested:", min(N, len(test_images)))

# =========================
# TEST LOOP
# =========================
correct = 0
tested_images = min(N, len(test_images))

for image_path in test_images[:tested_images]:

    # ---- Read image (COLOR or GRAYSCALE) ----
    img = cv2.imread(image_path)

    if img is None:
        continue

    # ---- Convert to grayscale if needed ----
    if len(img.shape) == 3:   # color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28, 28))

    # ---- Try to extract original label safely ----
    image_name = os.path.basename(image_path)
    original_label = None

    for ch in image_name:
        if ch.isdigit():
            original_label = int(ch)
            break

    # ---- Preprocess (same as training) ----
    img_norm = img.astype("float32") / 255.0
    img_norm = img_norm.reshape(1, -1)

    # ---- Predict ----
    prediction = model.predict(img_norm, verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # ---- Accuracy count (only if label exists) ----
    if original_label is not None and predicted_label == original_label:
        correct += 1

    # ---- Show image ----
    if SHOW_IMAGES:
        plt.imshow(img, cmap="gray")
        title = f"Predicted: {predicted_label} | Confidence: {confidence:.2f}%"
        if original_label is not None:
            title = f"Original: {original_label} | " + title

        plt.title(title)
        plt.axis("off")
        plt.show()

    # ---- Print result ----
    print(f"Image: {image_name}")
    print(f"   Original Label : {original_label}")
    print(f"   Predicted Label: {predicted_label}")
    print(f"   Confidence     : {confidence:.2f}%")
    print("-" * 40)

# =========================
# FINAL ACCURACY
# =========================
if tested_images > 0:
    overall_accuracy = (correct / tested_images) * 100
    print(f"\n✅ Overall Accuracy on tested images: {overall_accuracy:.2f}%")
else:
    print("❌ No images were tested.")
