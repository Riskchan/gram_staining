import sys

def usage():
    print("Usage: {0} <input_filename>".format(sys.argv[0]), file=sys.stderr)
    exit(1)

# Filename
if len(sys.argv) != 2:
    usage()
input_filename = sys.argv[1]

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Parameters
classes = ["Escherichia coli", "Staphylococcus aureus"]
num_classes = len(classes)
img_width, img_height = 128, 128
feature_dim = (img_width, img_height, 3)

# Load model
model = tf.keras.models.load_model("weights/vgg16-finetuning30.hdf5")

# Load image
img = image.load_img(input_filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# [0, 1] transformation
x = x / 255.0

# Prediction
pred = model.predict(x)[0]
print("{}: {}%".format(classes[0], (1-pred) * 100.0))
print("{}: {}%".format(classes[1], pred * 100.0))