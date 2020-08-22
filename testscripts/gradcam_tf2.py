import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def grad_cam(model, x, layer, width, height):
    """
    Args: 
        model(object): モデルオブジェクト
        x(ndarray): 画像
        layer(string): 畳み込み層の名前
        width, height: 画像のサイズ
    Returns:
        output_image(ndarray): 元の画像に色付けした画像
    """
    img = np.expand_dims(x, axis=0)
    img = img / 255.0

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(layer)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        iterate.layers[-1].activation = tf.keras.activations.linear
        model_out, last_conv_layer = iterate(img)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap[0], (width, height), cv2.INTER_LINEAR)

    return heatmap


def guided_grad_cam(input_model, x, layer_name, img_width, img_height):
    """
    Args: 
        input_model(object): モデルオブジェクト
        x(ndarray): 画像
        layer_name(string): 畳み込み層の名前
        img_width, img_height: 画像のサイズ
    Returns:
        output_image(ndarray): 元の画像に色付けした画像
    """

    # 画像の前処理
    # 読み込む画像が1枚なため、次元を増やしておかないとmode.predictが出来ない
    IMAGE_SIZE = (img_width, img_height)
    X = np.expand_dims(x, axis=0)
    preprocessed_input = X.astype('float32') / 255.0    

    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])
    grad_model.layers[-1].activation = tf.keras.activations.linear

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # 勾配を計算
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    # 重みを平均化して、レイヤーの出力に乗じる
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, IMAGE_SIZE, cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # ヒートマップを計算
    heatmap = cam / cam.max()

    return heatmap
