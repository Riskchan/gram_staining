import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def usage():
    print("Usage: {0} <input_filename>".format(sys.argv[0]), file=sys.stderr)
    exit(1)

if len(sys.argv) != 2:
    usage()
filename = sys.argv[1]

base_dir = "./"
data_dir = base_dir + "images"

# Tensorflow Parameters
classes = [filename for filename in os.listdir(data_dir) if not filename.startswith('.')]
num_classes = len(classes)
img_width, img_height = 256, 256
feature_dim = (img_width, img_height, 3)

# Load model
model = tf.keras.models.load_model("InceptionResNetV2/weights-InceptionResNetV2.hdf5")

# Crop size
crop_width, crop_height = 512, 512

img = Image.open(filename)
width, height = img.size
n_width = width // crop_width
n_height = height // crop_height

# Plot original figure
fig = plt.figure(figsize=(6.0,8.0))
ax = fig.add_subplot(2, 1, 1)
ax.axis("off")
ax.imshow(img)

# Plot prediction
ax = fig.add_subplot(2, 1, 2)
ax.axis("off")
copy_img = img.copy()
for i in range(n_width):
    for j in range(n_height):
        img_crop = copy_img.crop((i*crop_width, j*crop_height, (i+1)*crop_width, (j+1)*crop_height))
        img_crop = img_crop.resize((img_width, img_height))

        # Tensorflow settings
        x = image.img_to_array(img_crop)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Prediction
        pred = model.predict(x)[0]
        class_idx = np.argmax(pred)

        if class_idx == 0:
            overlay = Image.new("RGBA", (crop_width, crop_height), (255, 255, 255, 0))
        elif class_idx == 1:
            overlay = Image.new("RGBA", (crop_width, crop_height), (255, 0, 0, 50))
        elif class_idx == 2:
            overlay = Image.new("RGBA", (crop_width, crop_height), (0, 0, 255, 50))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, crop_width-1, crop_height-1), outline = (0, 0, 0))
        copy_img.paste(overlay, (i*crop_width, j*crop_height), overlay)
ax.imshow(copy_img)

plt.show()