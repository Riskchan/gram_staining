import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()

# 画像をロード
import urllib
img_src1 = "https://upload.wikimedia.org/wikipedia/commons/f/ff/Cristiano_Ronaldo_Euro2012_training_01.jpg"
img_src2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Eurasian_tree_sparrows_feeding_from_the_hand_in_Ueno_Park%2C_Tokyo%2C_Japan.jpg/480px-Eurasian_tree_sparrows_feeding_from_the_hand_in_Ueno_Park%2C_Tokyo%2C_Japan.jpg"

img_path1 = 'input_image1.jpg'
img_path2 = 'input_image2.jpg'
urllib.request.urlretrieve(img_src1, img_path1)
urllib.request.urlretrieve(img_src2, img_path2)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tensorflow.keras.preprocessing.image import load_img

# 画像をロード
img1 = load_img('input_image1.jpg', target_size=(224, 224))
img2 = load_img('input_image2.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2)])

# データセットの加工
X = preprocess_input(images)

# 画像を表示
subprot_args = {
   'nrows': 1,
   'ncols': 2,
   'figsize': (6, 3),
   'subplot_kw': {'xticks': [], 'yticks': []}
}
#f, ax = plt.subplots(**subprot_args)
#for i in range(len(images)):
#   ax[i].imshow(images[i])
#plt.tight_layout()
#plt.show()

# loss functionの設定
def loss(output):
   return (output[0][1], output[1][294])

# Define modifier to replace a softmax function of the last layer to a linear function.
def model_modifier(m):
   m.layers[-1].activation = tf.keras.activations.linear
   return m

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.utils import normalize

# Create Gradcam object
#gradcam = Gradcam(model, model_modifier, clone=False)
gradcam = GradcamPlusPlus(model, model_modifier, clone=False)
#gradcam = ScoreCAM(model, model_modifier, clone=False)

# Generate heatmap with GradCAM
cam = gradcam(loss, X)
cam = normalize(cam)

f, ax = plt.subplots(**subprot_args)
for i in range(len(cam)):
   ax[i].imshow(images[i])
   ax[i].imshow(cam[i], cmap='jet', alpha=0.5)
   plt.tight_layout()
plt.show()