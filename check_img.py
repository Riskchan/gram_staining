import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Parameters
batch_size = 32
epochs = 30
classes = ["GNR", "GPC"]
num_classes = len(classes)
img_width, img_height = 128, 128
feature_dim = (img_width, img_height, 3)
data_dir = "./images"

# Image data generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=90,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.3
)

iterator = datagen.flow_from_directory(
    data_dir,
    save_to_dir="processed",
    subset="validation",
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes=classes,
    class_mode="binary", #dog or cat-> binary, if more, set "categorical"
    batch_size=batch_size,
    shuffle=True
)
next(iterator)

