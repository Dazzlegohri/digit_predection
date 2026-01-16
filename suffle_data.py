#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil

SOURCE_FOLDER = "/Users/dazzle/Downloads/mnist-images/0-9 test images/0-9 test images"
OUTPUT_FOLDER = "/Users/dazzle/Downloads/mnist-images/0-9 test images/shuffled_data"

def shuffle_pngs_to_new_folder(src, dst):
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.lower().endswith(".png")]
    random.shuffle(files)

    for filename in files:
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)  # 🔥 SAME filename

        shutil.copy2(src_path, dst_path)

    print(f"✅ Shuffled {len(files)} PNG images into: {dst}")

shuffle_pngs_to_new_folder(SOURCE_FOLDER, OUTPUT_FOLDER)
