import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

def image_classifier(img, crop_width, crop_height, verbose=True):

    ver = "InceptionResNetV2_20201011"
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
    img_width, img_height = 512, 512
    feature_dim = (img_width, img_height, 3)

    # Image operation
    width, height = img.size
    n_width = width // crop_width
    n_height = height // crop_height

    result = []

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
            pred = model.predict(x)[0]
            sub_summary.append(pred)
            if verbose:
                print("################# Evaluating region ({:02d}, {:02d})... #################".format(i, j))
                for cls, prob in zip(classes, pred):
                    print("{0:50}{1:8.4f}%".format(cls, prob * 100.0))
        result.append(sub_summary)
    
    return class_indices, result

def calc_overall_probability(class_indices, result, max_method=False, max_key="", verbose=True):
    # Classes and indices
    classes = list(class_indices.keys())
    background_idx = class_indices["Background"]

    # Summary
    summary = dict.fromkeys(class_indices)
    for key in summary.keys():
        summary[key] = []

    for ss in result:
        for s in ss:
            class_idx = np.argmax(s)

            # Summary
            if class_idx != background_idx:
                for cls, prob in zip(classes, s):
                    summary[cls].append(prob*100)

    overall_prob = dict.fromkeys(class_indices)

    if max_method:
        idx = np.argmax(summary[max_key])
        for key in overall_prob.keys():
            overall_prob[key] = summary[key][idx]
    else:
        for key in overall_prob.keys():
            overall_prob[key] = sum(summary[key])/len(summary[key])

    if verbose:
        print("########################### Summary #############################")
        for key in overall_prob.keys():
            print("{0:50}{1:8.4f}%".format(key, overall_prob[key]))

    return overall_prob
