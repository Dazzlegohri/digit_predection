import os
from glob import glob
import cv2
import matplotlib.pyplot as plt

images_path = "/Users/dazzle/Downloads/mnist-images/0-9 traning images/0-9 traning images"

images = sorted(
    glob(os.path.join(images_path, "*.jpg")) +
    glob(os.path.join(images_path, "*.png"))
)

print("Total images:", len(images))

image_matrices = []
labels = []

for image in images[0:100]:
    curr_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image_matrices.append(curr_img)

    image_name = os.path.basename(image)
    first_char = image_name[0]
    labels.append(first_char)

    plt.imshow(curr_img, cmap="gray")
    plt.title(f"Label: {first_char}")
    plt.axis("off")
    plt.show()

    print("Image name:", image_name, " First character:", first_char)

print("\nSummary:")
print("Total loaded images:", len(image_matrices))
print("Labels:", labels)
print("Image size:", image_matrices[0].shape)
