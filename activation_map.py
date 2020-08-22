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

#tf.compat.v1.disable_eager_execution()

def grad_cam3(input_model, image, cls, layer_name):

    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    output, grads_val =  K.function([input_model.input], [conv_output, grads])([image])
    print('output.shape=', output.shape)
    print('grads_val.shape=', grads_val.shape)
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    print('weights.shape=', weights.shape)
    cam = np.dot(output, weights)
    print('cam.shape=', cam.shape)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    print('cam_new.shape=', cam.shape)
    cam = np.maximum(cam, 0)

    cam = cam / cam.max()
    return cam

def grad_cam2(input_model, x, layer_name, img_width, img_height):
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

    # モノクロ画像に疑似的に色をつける
#    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    # RGBに変換
#    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    # もとの画像に合成
#    output_image = (np.float32(rgb_cam) + x / 2)  

    return heatmap

## Grad-CAM function
 
def grad_cam(model, x, layer_name):
    """Grad-CAM function"""
    
    cls = np.argmax(model.predict(x))
    
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
 
    # Get outputs and grads
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([x])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    weights = np.mean(grads_val, axis=(0, 1)) # Passing through GlobalAveragePooling
 
    cam = np.dot(output, weights) # multiply
    cam = np.maximum(cam, 0)      # Passing through ReLU
    cam /= np.max(cam)            # scale 0 to 1.0
 
    return cls, cam

## Grad-CAM++ function
def grad_cam_plus_plus(model, x, layer_name):
    """Grad-CAM++ function"""
    
    cls = np.argmax(model.predict(x))
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
 
    first = K.exp(y_c) * grads
    second = K.exp(y_c) * grads * grads
    third = K.exp(y_c) * grads * grads * grads
 
    gradient_function = K.function([model.input], [y_c, first, second, third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = gradient_function([x])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)
 
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom
 
 
    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
    alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)
 
    cam = np.sum(deep_linearization_weights * conv_output[0], axis=2)
    cam = np.maximum(cam, 0) # Passing through ReLU
    cam /= np.max(cam)       # scale 0 to 1.0  
 
    return cls, cam

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
#model.summary()

plt.figure()
plt.tight_layout()
i = 1
for input_filename in input_filenames:    
    img = img_to_array(load_img(input_filename, target_size=(img_height, img_width)))
    plt.subplot(len(input_filenames), 2, i)
    plt.axis("off")
    plt.imshow(array_to_img(img))
    i += 1
    
    # Generate heatmap with GradCAM
    target_layer = "conv_7b_ac"
    x = np.expand_dims(img, axis=0)
    x = x / 255.0

    heatmap = grad_cam2(model, img, target_layer, img_width, img_height)
#    cls, cam = grad_cam(model, x, target_layer)
#    cls, cam = grad_cam_plus_plus(model, x, target_layer)
#    print(cam)
#    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
#    heatmap = np.uint8(255 * heatmap)
#    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    plt.subplot(len(input_filenames), 2, i)
    plt.axis("off")
    plt.imshow(array_to_img(img))
#    plt.imshow(heatmap)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    i += 1

plt.show()
#plt.savefig("heatmap.png")

