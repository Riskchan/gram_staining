import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from keras import backend as K
from tensorflow.keras import models
from keras.preprocessing.image import array_to_img, img_to_array, load_img

input_filenames = sys.argv[1:]

base_dir = "./"
data_dir = base_dir + "images"

# Parameters
classes = [filename for filename in os.listdir(data_dir) if not filename.startswith('.')]
num_classes = len(classes)
img_width, img_height = 256, 256
feature_dim = (img_width, img_height, 3)

# Load model
model = tf.keras.models.load_model("InceptionResNetV2/weights-InceptionResNetV2.hdf5")
model.summary()

from testscripts import gradcam_tf2

plt.figure()
plt.tight_layout()
i = 1
for input_filename in input_filenames:    
    img = img_to_array(load_img(input_filename, target_size=(img_height, img_width)))

    plt.subplot(len(input_filenames), 4, i)
    plt.axis("off")
    plt.imshow(array_to_img(img))
    plt.tight_layout()
    i += 1
    
    # Generate heatmap with GradCAM
    target_layer = "conv_7b_ac"

    # guided gradcam
    plt.subplot(len(input_filenames), 4, i)
    plt.axis("off")
    heatmap = gradcam_tf2.guided_grad_cam(model, img, target_layer, img_width, img_height)
    plt.imshow(array_to_img(img))
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    i += 1

plt.show()
#plt.savefig("heatmap.png")

