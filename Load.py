import cv2
import numpy as np
import pathlib
import datetime
from PIL import Image

import traceback

import Data as d

def load_train():
    valid = 0

    text = np.loadtxt(d.text_address, delimiter=" ")
    data = np.empty((d.num_train, d.num_rows, d.num_cols, d.channel), dtype="float64")
    label = np.empty(d.num_train, dtype="float64")

    for i in range(d.num_train):
        if text[i, 1] >= d.least_rate:
            if text[i, 3] <= d.large_border:
                img_address = d.image_address_train + str(int(text[i, 0])) + ".jpg"

                try:
                    if pathlib.Path(img_address).is_file():
                        input_image = cv2.imread(img_address)
                        input_image = d.fix_image_size(input_image)

                        #data[valid, :, :, 0] = d.get_score_list(input_image)
                        #data[valid, :, :, 1] = d.get_color_list(input_image)

                        data[valid, :, :, :] = d.get_hsv_list(input_image)

                        '''
                        max = d.find_max(d.get_score(input_image, (d.num_rows / 3) + 1, (d.num_cols / 3) + 1),
                                         d.get_score(input_image, (d.num_rows / 3) + 1, d.num_cols - (d.num_cols / 3) + 1),
                                         d.get_score(input_image, d.num_rows - (d.num_rows / 3) + 1, (d.num_cols / 3) + 1),
                                         d.get_score(input_image, d.num_rows - (d.num_rows / 3) + 1, d.num_cols - (d.num_cols / 3) + 1))
                        data[valid, :, :, 0] = max[1]
                        '''

                        label[valid] = d.classify(text[i, 2])
                        valid += 1

                        if valid % d.display_step == 0:
                            date = datetime.datetime.now()
                            date = date.strftime('%Y-%m-%d %H:%M:%S')
                            print(str(valid) + " data is loaded! (" + str(valid) + "/" + str(d.num_train) + ") " + date)

                except Exception:
                    continue

    data = np.resize(data, (valid, d.num_rows, d.num_cols, d.channel))

    return data, label, valid

def load_test():
    valid = 0

    text = np.loadtxt(d.text_address, delimiter=" ")
    data = np.empty((d.num_test, d.num_rows, d.num_cols, d.channel), dtype="float64")
    label = np.empty(d.num_test, dtype="float64")

    for i in range(d.num_test):
        if text[i, 1] >= d.least_rate:
            if text[i, 3] <= d.large_border:
                img_address = d.image_address_test + str(int(text[i + d.num_train, 0])) + ".jpg"

                try:
                    if pathlib.Path(img_address).is_file():
                        input_image = cv2.imread(img_address)
                        input_image = d.fix_image_size(input_image)

                        #data[valid, :, :, 0] = d.get_score_list(input_image)
                        #data[valid, :, :, 1] = d.get_color_list(input_image)

                        data[valid, :, :, :] = d.get_hsv_list(input_image)

                        '''
                        max = d.find_max(d.get_score(input_image, (d.num_rows / 3) + 1, (d.num_cols / 3) + 1),
                                         d.get_score(input_image, (d.num_rows / 3) + 1, d.num_cols - (d.num_cols / 3) + 1),
                                         d.get_score(input_image, d.num_rows - (d.num_rows / 3) + 1, (d.num_cols / 3) + 1),
                                         d.get_score(input_image, d.num_rows - (d.num_rows / 3) + 1, d.num_cols - (d.num_cols / 3) + 1))
                        data[valid, :, :, 0] = max[1]
                        '''

                        label[valid] = d.classify(text[i, 2])
                        valid += 1

                        if valid % d.display_step == 0:
                            date = datetime.datetime.now()
                            date = date.strftime('%Y-%m-%d %H:%M:%S')
                            print(str(valid) + " data is loaded! (" + str(valid) + "/" + str(d.num_test) + ") " + date)

                except Exception:
                    continue

    data = np.resize(data, (valid, d.num_rows, d.num_cols, d.channel))

    return data, label, valid

'''

text = np.loadtxt("data/rbf.txt", delimiter=" ")

total = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0

for i in range(10000):
    if text[i, 1] >= d.least_rate:
        if text[i, 3] <= d.large_border:
            img_address = d.image_address_test + str(int(text[i, 0])) + ".jpg"
            if pathlib.Path(img_address).is_file():
                img = Image.open(img_address)

                total += 1
                if d.classify(text[i, 2]) == 0:
                    c1 += 1
                if d.classify(text[i, 2]) == 1:
                    c2 += 1
                if d.classify(text[i, 2]) == 2:
                    c3 += 1
                if d.classify(text[i, 2]) == 3:
                    img.save('data/image/' + str(int(text[i, 0])) + "_" + str(int(text[i, 1])) + "_" + str(text[i, 2]) + ".jpg")
                    c4 += 1

print float(c1) / float(total)
print float(c2) / float(total)
print float(c3) / float(total)
print float(c4) / float(total)

'''