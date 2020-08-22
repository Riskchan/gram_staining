import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, InceptionV3, InceptionResNetV2, EfficientNetB0
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Base model for fine tuning
def get_base_model(modelname, input_shape):
    if modelname == "InceptionResNetV2":
        # InceptionResNetV2
        base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
        for layer in base_model.layers[:775]:
            layer.trainable = False

    elif modelname == "EfficientNetB0":
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
        for layer in base_model.layers[-20:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

    elif modelname == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)
        for layer in base_model.layers[:249]:
            layer.trainable = False
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True

        for layer in base_model.layers[249:]:
            layer.trainable = True

    elif modelname == "VGG16":
        base_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
        for layer in base_model.layers[:15]:
            layer.trainable = False

    elif modelname == "SmallCNN":
        layer_input = Input(shape=input_shape)

        layer_output = Conv2D(32, (3,3), activation="relu")(layer_input)
        layer_output = MaxPooling2D(pool_size=(2,2))(layer_output)

        layer_output = Conv2D(32, (3,3), activation="relu")(layer_output)
        layer_output = MaxPooling2D(pool_size=(2,2))(layer_output)

        layer_output = Conv2D(64, (3,3), activation="relu")(layer_output)
        layer_output = MaxPooling2D(pool_size=(2,2))(layer_output)

        base_model = training.Model(layer_input, layer_output, name='smallcnn')
    else:
        print("Unknown model name")
        exit(1)
    
    return base_model

# Project name and directories
# Choose one from following  SmallCNN, VGG16, InceptionV3, InceptionResNetV2, EfficientNetB0
ver = "InceptionResNetV2"

#base_dir = "./drive/My Drive/" # for Google colab
base_dir = "./"                 # for local use
data_dir = base_dir + "images"
ver_dir = base_dir + ver
processed_dir = ver_dir + "/processed"

os.makedirs(processed_dir, exist_ok=True)

# Parameters
batch_size = 32
epochs = 30
classes = [filename for filename in os.listdir(data_dir) if not filename.startswith('.')]
num_classes = len(classes)
img_width, img_height = 256, 256
feature_dim = (img_width, img_height, 3)

# Image data generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.3
)

train_generator = datagen.flow_from_directory(
    data_dir,
    subset="training",
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes=classes,
    class_mode="binary", #dog or cat-> binary, if more, set "categorical"
    batch_size=batch_size,
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    subset="validation",
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes=classes,
    class_mode="binary", # dog or cat-> binary, if more, set "categorical"
    batch_size=batch_size,
    shuffle=True
)

# Model definition
base_model = get_base_model(ver, feature_dim)
layer_output = base_model.output
layer_output = Flatten()(layer_output)
layer_output = Dense(256, activation="relu")(layer_output)
layer_output = Dropout(0.5)(layer_output)
layer_output = Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(.0005))(layer_output)
# If class_mode="binary", set 1 to Dense layer
# If class_mode="categorical", set num_classes to Dense layer

model = Model(base_model.input, layer_output)
model.summary()
model.compile(
    loss="binary_crossentropy",
#    optimizer=SGD(lr=1e-4, momentum=0.9),
    optimizer="Adam",
    metrics=["accuracy"]
)

# Training
cp_cb = ModelCheckpoint(
    filepath=ver_dir + "/" + "weights-" + ver + ".hdf5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode="min"
)

es_cb = EarlyStopping(
    monitor='val_loss', 
    patience=2, 
    verbose=1, 
    mode='auto'
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    verbose=1
)

num_train_samples = train_generator.n
num_validation_samples = validation_generator.n
steps_per_epoch_train = (num_train_samples-1) // batch_size + 1
steps_per_epoch_validation  = (num_validation_samples-1) // batch_size + 1
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    callbacks=[cp_cb, reduce_lr_cb]
)

# Trend line
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim(0, 1)
ax1.set_ylabel("accuracy")
rng = range(1, len(history.history["accuracy"]) + 1)
ax1.plot(rng, history.history["accuracy"], label="acc", ls="-", marker="o")
ax1.plot(rng, history.history["val_accuracy"], label="val_acc", ls="-", marker="x")

ax2 = ax1.twinx()
ax2.set_ylabel("loss")
ax2.plot(rng, history.history["loss"], label="loss", ls="-", marker="+")
ax2.plot(rng, history.history["val_loss"], label="val_loss", ls="-", marker="*")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='best')

plt.xlabel("epoch")
plt.savefig(ver_dir + "/accuracy.png")

print(train_generator.class_indices)
