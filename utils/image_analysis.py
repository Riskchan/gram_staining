import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

def image_classifier(img, crop_width, crop_height):

    ver = "InceptionResNetV2_20201003"
    base_dir = "./"
    ver_dir = base_dir + ver

    # Load model and class indices
    model_path = os.path.join(ver_dir, "weights-InceptionResNetV2.hdf5")
    model = tf.keras.models.load_model(model_path)

    class_indices_path = os.path.join(ver_dir, 'class_indices.pickle')
    with open(class_indices_path,'rb') as f:
        class_indices = pickle.load(f)
    classes = list(class_indices.keys())

    # Sizes of tensorflow input
    img_width, img_height = 256, 256
    feature_dim = (img_width, img_height, 3)

    # Image operation
    width, height = img.size
    n_width = width // crop_width
    n_height = height // crop_height

    summary = []

    # Plot prediction
    copy_img = img.copy()
    for i in range(n_width):
        sub_summary = []
        for j in range(n_height):
            img_crop = copy_img.crop((i*crop_width, j*crop_height, (i+1)*crop_width, (j+1)*crop_height))
            img_crop = img_crop.resize((img_width, img_height))

            # Tensorflow settings
            x = image.img_to_array(img_crop)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # Prediction
            print("################# Evaluating region ({:02d}, {:02d})... #################".format(i, j))
            pred = model.predict(x)[0]
            sub_summary.append(pred)
            for cls, prob in zip(classes, pred):
                print("{0:50}{1:8.4f}%".format(cls, prob * 100.0))
        summary.append(sub_summary)
    
    return class_indices, summary
