import os
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from utils import image_analysis

def usage():
    print("Usage: {0} <input_filename>".format(sys.argv[0]), file=sys.stderr)
    exit(1)

if len(sys.argv) != 2:
    usage()
filename = sys.argv[1]

# Load image
img = Image.open(filename)

# Image classification
crop_width, crop_height = 512, 512
class_indices, result = image_analysis.image_classifier(img, crop_width, crop_height)

# Calculate overall probability
#overall_prob = image_analysis.calc_overall_probability(class_indices, result, max_method=True, max_key="Staphylococcus aureus")
overall_prob = image_analysis.calc_overall_probability(class_indices, result)

# Classes and indices
classes = list(class_indices.keys())
background_idx = class_indices["Background"]
nonbg_class_idx = list(class_indices.values())
nonbg_class_idx.remove(background_idx)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis("off")
copy_img = img.copy()

i = 0
for ss in result:
    j = 0
    for s in ss:
        class_idx = np.argmax(s)

        # Categorization
        if class_idx == background_idx:
            overlay = Image.new("RGBA", (crop_width, crop_height), (255, 255, 255, 0))
        elif class_idx == nonbg_class_idx[0]:
            overlay = Image.new("RGBA", (crop_width, crop_height), (255, 0, 0, 50))
        elif class_idx == nonbg_class_idx[1]:
            overlay = Image.new("RGBA", (crop_width, crop_height), (0, 0, 255, 50))

        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, crop_width-1, crop_height-1), outline = (0, 0, 0))
        copy_img.paste(overlay, (i*crop_width, j*crop_height), overlay)
        j += 1
    i += 1

width, height = img.size
back = Image.new("RGB", (width, height+1200), (255, 255, 255))
back.paste(copy_img, (0, 0))

# Legend
classes_legend = classes.copy()
classes_legend.remove("Background")

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

draw = ImageDraw.Draw(legend_area)
font = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 200)
draw.text((rec_size + padding, padding), classes_legend[0], font=font, fill="black")
draw.text((rec_size + padding, rec_size + 2*padding), classes_legend[1], font=font, fill="black")
back.paste(legend_area, (0, height))
ax.imshow(back)

plt.show()