#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:50:39 2026

@author: dazzle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Digit Recognition using File Picker
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import os

# ============================
# Load Model
# ============================
model = load_model("digit_model.h5")
print(" Model loaded successfully")

# ============================
# File Picker Dialog
# ============================
root = tk.Tk()
root.withdraw()  # Hide main tkinter window

print(" Please select an image...")
image_path = filedialog.askopenfilename(
    title="Select Digit Image",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
)

if not image_path:
    raise ValueError(" No image selected.")

print("📸 Selected Image:", image_path)

# ============================
# Read & Preprocess Image
# ============================
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(" Unable to read image.")

img_resized = cv2.resize(img, (28, 28))

img_norm = img_resized.astype("float32") / 255.0
img_norm = img_norm.reshape(1, -1)

# ============================
# Prediction
# ============================
prediction = model.predict(img_norm, verbose=0)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

# ============================
# Show Image & Result
# ============================
plt.imshow(img_resized, cmap="gray")
plt.title(
    f"Predicted Digit: {predicted_digit}\n"
    f"Confidence: {confidence:.2f}%"
)
plt.axis("off")
plt.show()

# ============================
# Print Output
# ============================
print("\n Prediction Result")
print("---------------------")
print(f"🧠 Predicted Digit : {predicted_digit}")
print(f"📊 Confidence      : {confidence:.2f}%")
