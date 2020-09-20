import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

input_filenames = sys.argv[1:]
crop_dir = "./cropped"
os.makedirs(crop_dir, exist_ok=True)

crop_width, crop_height = 512, 512

for filename in input_filenames:
    name = filename.split("/")[-1].split(".")[0]
    img = Image.open(filename)
    width, height = img.size
    n_width = width // crop_width
    n_height = height // crop_height
    
    for i in range(n_width):
        for j in range(n_height):
            img_crop = img.crop((i*crop_width, j*crop_height, (i+1)*crop_width, (j+1)*crop_height))
            img_crop.save("{}/{}-{:03d}x{:03d}.png".format(crop_dir, name, i, j))
