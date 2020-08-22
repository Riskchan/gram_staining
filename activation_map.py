import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.utils import normalize

# The `output` variable refer to the output of the model,
# so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
def loss(output):
#    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
#    return (output[0][1], output[1][294], output[2][413])
    return (output[0][0])

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

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

# Grad-CAM calculation
gradcam = Gradcam(model, model_modifier, clone=False)
#gradcam = GradcamPlusPlus(model, model_modifier, clone=False)

plt.figure()
plt.axis("off")
plt.tight_layout()
i = 1
for input_filename in input_filenames:    
    img = img_to_array(load_img(input_filename, target_size=(img_height, img_width)))
    plt.subplot(len(input_filenames), 2, i)
    plt.axis("off")
    plt.imshow(array_to_img(img))
    i += 1
    
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)

    # [0, 1] transformation
#    x = x / 255.0

    # Generate heatmap with GradCAM
    cam = gradcam(loss, img, penultimate_layer=-1)
    cam = normalize(cam)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)

    plt.subplot(len(input_filenames), 2, i)
    plt.axis("off")
    plt.imshow(array_to_img(img))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    i += 1

plt.show()
#plt.savefig("heatmap.png")

