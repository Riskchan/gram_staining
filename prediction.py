import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

input_filenames = sys.argv[1:]

base_dir = "./"
data_dir = base_dir + "images"

# Parameters
class_indices_path = os.path.join(ver_dir, 'class_indices.pickle')
with open(class_indices_path,'rb') as f:
    class_indices = pickle.load(f)
classes = list(class_indices.keys())
num_classes = len(classes)
img_width, img_height = 256, 256
feature_dim = (img_width, img_height, 3)

# Load model
model = tf.keras.models.load_model("InceptionResNetV2/weights-InceptionResNetV2.hdf5")

# Load image
for input_filename in input_filenames:
    img = image.load_img(input_filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # [0, 1] transformation
    x = x / 255.0

    # Prediction
    pred = model.predict(x)[0]

    # Prediction
    print("{}".format(input_filename))
    if num_classes == 2:
        print("{}: {}%".format(classes[0], (1-pred) * 100.0))
        print("{}: {}%".format(classes[1], pred * 100.0))
    else:
        for cls, prob in zip(classes, pred):
            print("{0:18}{1:8.4f}%".format(cls, prob * 100.0))