import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from keras import backend as K
from tensorflow.keras import models
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import gradcam

input_filenames = sys.argv[1:]

ver = "InceptionResNetV2"

base_dir = "./"
ver_dir = base_dir + ver
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
model.summary()


fig = plt.figure(figsize=(6.0,8.0))
num_images = len(input_filenames)*2
cols = 4
rows = num_images//cols if num_images%cols == 0 else num_images//cols + 1
i = 1
for input_filename in input_filenames:    
    img = img_to_array(load_img(input_filename, target_size=(img_height, img_width)))

    # Prediction
    x = np.expand_dims(img, axis=0)
    x = x / 255.0
    pred = model.predict(x)
    pred_idx = np.argmax(pred[0])

    # Plot original image
    ax = fig.add_subplot(rows, cols, i)
    ax.axis("off")
    ax.imshow(array_to_img(img))
    ax.set_title("{}".format(classes[pred_idx]))
    i += 1
    
    # Generate heatmap with Guided GradCAM
    target_layer = "conv_7b_ac"
    ax = fig.add_subplot(rows, cols, i)
    ax.axis("off")
    heatmap = gradcam.guided_grad_cam(model, img, target_layer, img_width, img_height)
    ax.imshow(array_to_img(img))
    ax.imshow(heatmap, alpha=0.5, cmap='jet')
    i += 1

plt.show()
#plt.savefig("heatmap.png")

