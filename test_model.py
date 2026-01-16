#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:27:19 2026

@author: Dazzle
"""

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model


# model = load_model("digit_model.h5")
# print("✅ Model loaded")


# test_image_path = "/Users/dazzle/Downloads/mnist-images/0-9 test images/0-9 test images"   

# img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# if img is None:
#     raise ValueError("❌ Image not found. Check path.")

# img = cv2.resize(img, (28, 28))

# plt.imshow(img, cmap="gray")
# plt.title("Test Image")
# plt.axis("off")
# plt.show()


# img = img.astype("float32") / 255.0
# img = img.reshape(1, -1)   


# prediction = model.predict(img)
# predicted_digit = np.argmax(prediction)

# print("🎯 Predicted Digit:", predicted_digit)








import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


N = 100           # Images that u want to test
SHOW_IMAGES = True     # false if don't want images


model = load_model("digit_model.h5")
print(" Model loaded")


test_images_path = "/Users/dazzle/Downloads/mnist-images/0-9 test images/test_random"

test_images = sorted(
    glob(os.path.join(test_images_path, "*.png")) +
    glob(os.path.join(test_images_path, "*.jpg"))
)

if len(test_images) == 0:
    raise ValueError(" No test images found. Check folder path.")

print("Total test images available:", len(test_images))
print("Images tested:", min(N, len(test_images)))


correct = 0
tested_images = min(N, len(test_images))

for image_path in test_images[:tested_images]:


    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (28, 28))

    image_name = os.path.basename(image_path)
    original_label = int(image_name[0])

    img_norm = img.astype("float32") / 255.0
    img_norm = img_norm.reshape(1, -1)

    prediction = model.predict(img_norm, verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # %

    if predicted_label == original_label:
        correct += 1

    if SHOW_IMAGES:
        plt.imshow(img, cmap="gray")
        plt.title(
            f"Original: {original_label} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.2f}%"
        )
        plt.axis("off")
        plt.show()

    print(f"Image: {image_name}")
    print(f"   Original Label : {original_label}")
    print(f"   Predicted Label: {predicted_label}")
    print(f"   Confidence     : {confidence:.2f}%")
    print("-" * 40)


overall_accuracy = (correct / tested_images) * 100
print(f"\n Overall Accuracy on tested images: {overall_accuracy:.2f}%")

