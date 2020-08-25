import os
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def usage():
    print("Usage: {0} <input_filename>".format(sys.argv[0]), file=sys.stderr)
    exit(1)

if len(sys.argv) != 2:
    usage()
filename = sys.argv[1]

ver = "InceptionResNetV2"
base_dir = "./"
data_dir = base_dir + "images"
ver_dir = base_dir + ver

# Load model and class indices
model_path = os.path.join(ver_dir, "weights-{}.hdf5".format(ver))
model = tf.keras.models.load_model(model_path)

class_indices_path = os.path.join(ver_dir, 'class_indices.pickle')
with open(class_indices_path,'rb') as f:
    class_indices = pickle.load(f)
classes = list(class_indices.keys())
num_classes = len(classes)

background_idx = class_indices["Background"]
nonbg_class_idx = list(class_indices.values())
nonbg_class_idx.remove(background_idx)

# Sizes of tensorflow input
img_width, img_height = 256, 256
feature_dim = (img_width, img_height, 3)

# Crop size
crop_width, crop_height = 512, 512

# Image operation
img = Image.open(filename)
width, height = img.size
n_width = width // crop_width
n_height = height // crop_height

n_wloop = n_width if width % crop_width == 0 else n_width + 1
n_hloop = n_height if width % crop_height == 0 else n_height + 1

# Summary
num_non_ng = 0
summary = dict.fromkeys(class_indices)
for key in summary.keys():
    summary[key] = 0

# Plot original figure
fig = plt.figure()

# Plot prediction
ax = fig.add_subplot(1, 1, 1)
ax.axis("off")
copy_img = img.copy()
for i in range(n_wloop):
    for j in range(n_hloop):
        img_crop = copy_img.crop((i*crop_width, j*crop_height, (i+1)*crop_width, (j+1)*crop_height))
        img_crop = img_crop.resize((img_width, img_height))

        # Tensorflow settings
        x = image.img_to_array(img_crop)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Prediction
        print("################# Evaluating region ({:02d}, {:02d})... #################".format(i, j))
        pred = model.predict(x)[0]
        for cls, prob in zip(classes, pred):
            print("{0:50}{1:8.4f}%".format(cls, prob * 100.0))

        class_idx = np.argmax(pred)

        # Summary
        if class_idx != background_idx:
            num_non_ng += 1
            for cls, prob in zip(classes, pred):
                summary[cls] += prob*100.0

        # Categorization
        if i < n_width and j < n_height:
            if class_idx == background_idx:
                overlay = Image.new("RGBA", (crop_width, crop_height), (255, 255, 255, 0))
            elif class_idx == nonbg_class_idx[0]:
                overlay = Image.new("RGBA", (crop_width, crop_height), (255, 0, 0, 50))
            elif class_idx == nonbg_class_idx[1]:
                overlay = Image.new("RGBA", (crop_width, crop_height), (0, 0, 255, 50))
        else:
            overlay = Image.new("RGBA", (crop_width, crop_height), (0, 0, 0, 100))

        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, crop_width-1, crop_height-1), outline = (0, 0, 0))
        copy_img.paste(overlay, (i*crop_width, j*crop_height), overlay)

print("########################### Summary #############################")
for cls, prob in zip(classes, pred):
    summary[cls] /= num_non_ng
    print("{0:50}{1:8.4f}%".format(cls, summary[cls]))


back = Image.new("RGB", (width, height+1200), (255, 255, 255))
back.paste(copy_img, (0, 0))

# Legend
classes_legend = classes.copy()
classes_legend.remove("Background")
classes_legend.append("Not inspected")

rec_size = 200
padding = 100
legend_area = Image.new("RGBA", (width, 1200), (255, 255, 255, 255))

rect = Image.new("RGBA", (rec_size, rec_size), (255, 0, 0, 50))
draw = ImageDraw.Draw(rect)
draw.rectangle((0, 0, rec_size-1, rec_size-1), outline = (0, 0, 0))
legend_area.paste(rect, (0, padding), rect)

rect = Image.new("RGBA", (rec_size, rec_size), (0, 0, 255, 50))
draw = ImageDraw.Draw(rect)
draw.rectangle((0, 0, rec_size-1, rec_size-1), outline = (0, 0, 0))
legend_area.paste(rect, (0, rec_size+2*padding), rect)

rect = Image.new("RGBA", (rec_size, rec_size), (0, 0, 0, 100))
draw = ImageDraw.Draw(rect)
draw.rectangle((0, 0, rec_size-1, rec_size-1), outline = (0, 0, 0))
legend_area.paste(rect, (0, 2*rec_size+3*padding), rect)

draw = ImageDraw.Draw(legend_area)
font = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 200)
draw.text((rec_size + padding, padding), classes_legend[0], font=font, fill="black")
draw.text((rec_size + padding, rec_size + 2*padding), classes_legend[1], font=font, fill="black")
draw.text((rec_size + padding, 2*rec_size + 3*padding), classes_legend[2], font=font, fill="black")
back.paste(legend_area, (0, height))
ax.imshow(back)

plt.show()