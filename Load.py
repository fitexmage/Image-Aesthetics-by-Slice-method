from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import datetime
import numpy as np
import pathlib
import scipy
import subprocess

import Data as d

# Load text

def load_train():

    text = np.loadtxt(d.list_address_train, delimiter=" ")  # Make text data to array.

    valid = 0 # Number of valid image used for training.
    data = np.empty((5 * d.num_train, d.img_rows, d.img_cols, d.channel), dtype="float64") # Initialize an array for storing image data.
    label = np.empty(5 * d.num_train, dtype="float64") # Initialize an array for storing labels.

    # Read data and labels

    for i in range(d.num_train): # Iteration of all the image data
        #if text[random_image, 1] >= d.least_rate: # Keep the one which has more than a fixed number of people rating.
            #if text[random_image, 2] <= 4.5 or text[random_image, 2] >= 5.5:
                #if text[random_image, 3] <= d.large_border:
                    img_address = d.image_address_train + str(int(text[i, 0])) + ".jpg" # Define the image address.
                    img = load_img(img_address) # Load images.

                    '''
                    img = d.rapid(img)
                    data[i, :, :, :] = img
                    label[i] = d.classify(text[i, 1])

                    '''

                    img_1, img_2, img_3, img_4, img_5 = d.slice(img) # Slice images.
                    data[i, :, :, :] = img_1 # Add image to array.
                    data[i + d.num_train, :, :, :] = img_2
                    data[i + 2 * d.num_train, :, :, :] = img_3
                    data[i + 3 * d.num_train, :, :, :] = img_4
                    data[i + 4 * d.num_train, :, :, :] = img_5
                    label[i] = d.classify(text[i, 1]) # Add label to array.
                    label[i + d.num_train] = d.classify(text[i, 1])
                    label[i + 2 * d.num_train] = d.classify(text[i, 1])
                    label[i + 3 * d.num_train] = d.classify(text[i, 1])
                    label[i + 4 * d.num_train] = d.classify(text[i, 1])

                    valid += 5  # Add the count of valid images.


                    if valid % d.display_step == 0:
                        date = datetime.datetime.now()
                        date = date.strftime('%Y-%m-%d %H:%M:%S')
                        print(str(valid) + " data is loaded! (" + str(valid) + "/" + str(5 * d.num_train) + ") " + date) # Return the process.

    return data, label, valid


def load_test():

    text = np.loadtxt(d.list_address_test, delimiter=" ")

    valid = 0
    data = np.empty((d.num_test, d.img_rows, d.img_cols, d.channel),dtype="float64")
    label = np.empty(d.num_test, dtype="float64")

    for i in range(d.num_test):
        #if text[random_image, 1] >= d.least_rate:
            #if text[random_image, 2] <= 4.5 or text[random_image, 2] >= 5.5:
                #if text[random_image, 3] <= d.large_border:
                    img_address = d.image_address_test + str(int(text[i, 0])) + ".jpg"
                    img = load_img(img_address)

                    '''

                    img = d.rapid(img)
                    data[i, :, :, :] = img
                    label[i] = d.classify(text[i, 1])

                    '''

                    img_1, img_2, img_3, img_4, img_5 = d.slice(img)
                    data[i, :, :, :] = img_5
                    label[i] = d.classify(text[i, 1])

                    valid += 1

                    if valid % d.display_step == 0:
                        date = datetime.datetime.now()
                        date = date.strftime('%Y-%m-%d %H:%M:%S')
                        print(str(valid) + " data is loaded! (" + str(valid) + "/" + str(d.num_test) + ") " + date)

    return data, label, valid
