from keras.models import model_from_json
import numpy as np
import os
import pathlib
import cv2

import Data as d

json_file = open(d.model_address, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(d.parameters_address)
print("Model loaded!")

# Predict random images in training data

print("Predict test images in training data: ")

imgs = os.listdir("data/test")

for i in range(len(imgs)):
    if imgs[i].endswith("jpg"):
        img_address = "data/test/" + imgs[i]
        input_image = cv2.imread(img_address)
        input_image = d.fix_image_size(input_image)

        score = np.empty((1, d.num_rows, d.num_cols, d.channel), dtype="float64")
        score[0, :, :, 0] = d.get_score_list(input_image)
        #score[0, :, :, 1] = d.get_color_list(input_image)
        prediction = model.predict(score)
        print str(prediction)
        #print img_address, str(np.argmax(prediction))
