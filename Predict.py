from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import pathlib
import scipy
import subprocess

import Data as d

json_file = open(d.model_address, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(d.parameters_address)
print("Model loaded!")



# Predict random images in training data

print("Predict random images in training data: ")

text = np.loadtxt(d.text_address,delimiter = " ")
right = 0

for i in range(d.num_prediction):
    random_image = np.random.randint(0, len(text))
    if text[random_image, 1] >= d.least_rate:
        if text[random_image, 2] <= 4.5 or text[random_image, 2] >= 5.5:
            if text[random_image, 3] < 2:
                img_address = d.image_address_train + str(int(text[random_image, 0])) + ".jpg"
                if pathlib.Path(img_address).is_file():
                    try:
                        img = load_img(img_address)
                        img_1, img_2, img_3, img_4, img_5 = d.slice(img)
                    except Exception:
                        continue

                    img_5 = img_5.reshape(1, d.img_rows, d.img_cols, d.channel)

                    prediction = model.predict(img_5)
                    if np.argmax(prediction) == d.classify(text[random_image, 2]):
                        right += 1

print("Accuracy is " + "%.2f%%" % (float(right)/float(d.num_prediction)*100))

'''

# Predict given images in "prediction" file

print("Predict given images in 'prediction' file: ")

imgs = os.listdir(d.image_address_predict)

for i in range(len(imgs)):
    if imgs[i].endswith("jpg"):
        img_address = d.prediction_address + imgs[i]
        try:
            img = subprocess.check_output([d.extractor_address, img_address])
        except subprocess.CalledProcessError:
            print("\n" + imgs[i] + " cannot be used!")
            break
        img = np.fromstring(img, sep=' ')
        img = img.reshape(1, d.img_rows, d.img_cols, d.channel)
        prediction = model.predict(img)

        print("\nName: " + imgs[i] + "\n" +
              "Rate: " + d.convert_rate(np.argmax(prediction)) + "\n\n" +
              "Low: " + "  %.2f%%" % (prediction[0][0] * 100) + "\n" +
              "High: " + "  %.2f%%" % (prediction[0][1] * 100))

'''