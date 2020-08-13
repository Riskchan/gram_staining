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

print(train_generator.class_indices)


# Model definition
layer_input = Input(shape=feature_dim)

layer_output = Conv2D(32, (3,3), activation="relu")(layer_input)
layer_output = MaxPooling2D(pool_size=(2,2))(layer_output)

layer_output = Conv2D(32, (3,3), activation="relu")(layer_output)
layer_output = MaxPooling2D(pool_size=(2,2))(layer_output)

layer_output = Conv2D(64, (3,3), activation="relu")(layer_output)
layer_output = MaxPooling2D(pool_size=(2,2))(layer_output)

layer_output = Flatten()(layer_output)
layer_output = Dense(64, activation="relu")(layer_output)
layer_output = Dropout(0.5)(layer_output)
layer_output = Dense(1, activation="sigmoid")(layer_output)
# If class_mode="binary", set 1 to Dense layer
# If class_mode="categorical", set num_classes to Dense layer

model = Model(layer_input, layer_output)
model.summary()
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Training
cp_cb = ModelCheckpoint(
    filepath="weights/smallcnn.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5",
    monitor="val_loss",
    verbose=1,
    mode="auto"
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

# === 正解率の推移出力 ===
plt.plot(range(1, len(history.history["accuracy"]) + 1),
         history.history["accuracy"],
         label="acc", ls="-", marker="o")
plt.plot(range(1, len(history.history["val_accuracy"]) + 1),
         history.history["val_accuracy"],
         label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.savefig("accuracy.png")
plt.show()

