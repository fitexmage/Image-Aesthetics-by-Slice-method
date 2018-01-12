from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pathlib
import traceback

import Data as d

text = np.loadtxt(d.text_address, delimiter=" ")
num_list = 40000
display_step = 1000
left = 4.5
right = 5.5

def getRapidList():
    valid = 0
    i = 0
    while valid < num_list:
        i += 1
        if text[i, 1] >= d.least_rate:
            if text[i, 2] < left or text[i, 2] > right:
                if text[i, 3] <= d.large_border:
                    img_address = d.image_address_train + str(int(text[i, 0])) + ".jpg"

                    try:
                        if pathlib.Path(img_address).is_file():

                            img = load_img(img_address)
                            img = d.rapid(img)

                            if img.shape[0] != d.img_rows & img.shape[1] != d.img_cols:
                                continue

                            valid += 1

                            f = open("list.txt", 'a')
                            f.write(str(int(text[i, 0])) + " " + str(text[i, 2]) + "\n")
                            f.close()

                            if valid % display_step == 0:
                                print(str(valid) + "/" + str(num_list) + " data is loaded! (" + str(i) + "/" + str(len(text)) + ") ")

                    except Exception:
                        continue

def getSliceList():

    valid = 0
    i = 0
    while valid < num_list:
        i += 1
        if text[i, 1] >= d.least_rate:
            if text[i, 2] < left or text[i, 2] > right:
                if text[i, 3] <= d.large_border:
                    img_address = d.image_address_train + str(int(text[i, 0])) + ".jpg"

                    try:
                        if pathlib.Path(img_address).is_file():

                            img = load_img(img_address)
                            img_1, img_2, img_3, img_4, img_5 = d.slice(img)  # Slice images.

                            if img_1.shape[0] != d.img_rows | img_1.shape[1] != d.img_cols:
                                continue
                            elif img_2.shape[0] != d.img_rows | img_1.shape[1] != d.img_cols:
                                continue
                            elif img_3.shape[0] != d.img_rows | img_1.shape[1] != d.img_cols:
                                continue
                            elif img_4.shape[0] != d.img_rows | img_1.shape[1] != d.img_cols:
                                continue
                            elif img_5.shape[0] != d.img_rows | img_1.shape[1] != d.img_cols:
                                continue

                            valid += 1

                            f = open("list.txt", 'a')
                            f.write(str(int(text[i, 0])) + " " + str(text[i, 2]) + "\n")
                            f.close()

                            if valid % display_step == 0:
                                print(str(valid) + "/" + str(num_list) + " data is loaded! (" + str(i) + "/" + str(len(text)) + ") ")

                    except Exception:
                        continue

def test():
    count1 = 0
    count2 = 0
    text = np.loadtxt("list.txt", delimiter=" ")
    for i in range(len(text)):
        if text[i,1] < left:
            count1 += 1
        if text[i,1] > right:
            count2 += 1

    print count1
    print count2

#test()
getSliceList()
#getRapidList()

# middle_2000: 470000
# middle_40000: 0
