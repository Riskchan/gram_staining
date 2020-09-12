import os
import io
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

# Parameters
with open('weights/class_indices.pickle','rb') as f:
    class_indices = pickle.load(f)
classes = list(class_indices.keys())
num_classes = len(classes)
img_width, img_height = 256, 256
feature_dim = (img_width, img_height, 3)

background_idx = class_indices["Background"]
#nonbg_class_idx = list(class_indices.values())
#nonbg_class_idx.remove(background_idx)

# Load model
model = tf.keras.models.load_model("weights/weights-InceptionResNetV2.hdf5")

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        # Reject unexpected extensions
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>Extension not allowed</p> '''

        # Save/Load image
        img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(img_url)
        img = image.load_img(img_url)

        # Cropping
        crop_width, crop_height = 512, 512
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
        
        copy_img = img.copy()
        for i in range(n_wloop):
            for j in range(n_hloop):
                img_crop = copy_img.crop((i*crop_width, j*crop_height, (i+1)*crop_width, (j+1)*crop_height))
                img_crop = img_crop.resize((img_width, img_height))

                # [0, 1] transformation
                x = image.img_to_array(img_crop)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0

                # Prediction
                print("################# Evaluating region ({:02d}, {:02d})... #################".format(i, j))
                pred = model.predict(x)[0]
                pred = np.round(pred*100.0, decimals=2)
                result = zip(classes, pred)
                for cls, prob in result:
                    print("{0:50}{1:8.2f}%".format(cls, prob))

                class_idx = np.argmax(pred)

                # Summary
                if class_idx != background_idx:
                    num_non_ng += 1
                    for cls, prob in zip(classes, pred):
                        summary[cls] += prob


        print("########################### Summary #############################")
        for cls, prob in zip(classes, pred):
            summary[cls] /= num_non_ng
            summary[cls] = round(summary[cls], 2)
            print("{0:50}{1:8.2f}%".format(cls, summary[cls]))

        # Output results
        res = []
        for i in range(num_classes):
            res.append({'type': classes[i], 'prob': summary[classes[i]]})

        return render_template('index.html', img_url=img_url, result = res)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()