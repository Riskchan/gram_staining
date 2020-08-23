import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def usage():
    print("Usage: {0} <input_filename>".format(sys.argv[0]), file=sys.stderr)
    exit(1)

if len(sys.argv) != 2:
    usage()
filename = sys.argv[1]

crop_width, crop_height = 512, 512

img = Image.open(filename)
width, height = img.size
n_width = width // crop_width
n_height = height // crop_height

fig = plt.figure(figsize=(6.0,8.0))
ax = fig.add_subplot(2, 1, 1)
ax.axis("off")
ax.imshow(img)


ax = fig.add_subplot(2, 1, 2)
ax.axis("off")
copy_img = img.copy()
for i in range(n_width):
    for j in range(n_height):
        if i%2 == 0:
            class_img = Image.new("RGBA", (crop_width, crop_height), (255, 0, 0, 50))
        else:
            class_img = Image.new("RGBA", (crop_width, crop_height), (0, 255, 0, 50))

        copy_img.paste(class_img, (i*crop_width, j*crop_height), class_img)
ax.imshow(copy_img)

plt.show()