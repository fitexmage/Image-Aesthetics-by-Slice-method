from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import scipy

# Version

version = "0.4.9"

# Data for loading images

img_rows, img_cols, channel = 128, 128, 3 # The size of images after resizing and cropping.
num_train = 40000 # Number of training data
num_test = 2000 # Number of testing data
least_rate = 10 # The least times of rating
large_border = 2 # The largest border of images
display_step = 5000 # The display step for loading the data

# Address

text_address = "data/rbf.txt"
list_address_train = "data/train_list.txt"
list_address_test = "data/test_list.txt"
image_address_train = "/backups4/fuf111/DATASETs/photonet/original/"
image_address_test = "/backups4/fuf111/DATASETs/photonet/original/"
extractor_address = "data/aesth_feature_extractor_250.cc.x"
image_address_predict = "prediction/"
parameters_address = "data/model.h5"
model_address = "data/model.json"
result_address = "data/result.txt"
error_address = "data/error.txt"

# Data for model

batch_size = 64 # Batch size
num_classes = 2 # Number of classes
epochs = 10 # Epochs

# Data for prediction

num_prediction = 100 # Number of images for prediction

# Method

# Transfer y to categorical

def categorical(y, valid):
    c = np.zeros((valid,num_classes))
    for i in range(valid):
        c[i,int(y[i])] = 1
    return c

# Classify rating to different classes

def classify(rate):
    if rate >= 5:
        return 1
    elif rate < 5:
        return 0

# Scale image to [-1,1]

def scale(img):
    img /= 255
    img -= 0.5
    img *= 2.
    return img

# Make the larger side to 256, then slice images to five parts

def slice(img):
    resize_long = 256
    if img.size[0] < img.size[1]:
        if img.size[1] >= resize_long:
            resize_short = int(float(img.size[0]) / float(img.size[1]) * resize_long)
            if resize_short > (resize_long - img_rows):
                step = int((resize_short - img_cols) / (resize_long / img_rows - 1))
                img = img_to_array(img)
                img = scipy.misc.imresize(img, (resize_long, resize_short, channel))
                img = array_to_img(img)
                img_1 = img.crop((0, 0, img_cols, img_rows))
                img_2 = img.crop((step, 0, step + img_cols, img_rows))
                img_3 = img.crop((0, img_rows, img_cols, 2 * img_rows))
                img_4 = img.crop((step, img_rows, step + img_cols, 2 * img_rows))
                img_5 = img.crop((int((resize_short-img_cols)/2), int((resize_long-img_rows)/2), int((resize_short-img_cols)/2)+img_cols, int((resize_long-img_rows)/2)+img_rows))

    if img.size[0] >= img.size[1]:
        if img.size[0] >= resize_long:
            resize_short = int(float(img.size[1]) / float(img.size[0]) * resize_long)
            if resize_short > (resize_long - img_rows):
                step = int((resize_short - img_rows) / (resize_long / img_cols - 1))
                img = scipy.misc.imresize(img, (resize_short, resize_long, channel))
                img = array_to_img(img)
                img_1 = img.crop((0, 0, img_cols, img_rows))
                img_2 = img.crop((img_cols, 0, 2 * img_cols, img_rows))
                img_3 = img.crop((0, step, img_cols, step + img_rows))
                img_4 = img.crop((img_cols, step, 2 * img_cols, step + img_rows))
                img_5 = img.crop((int((resize_long - img_cols) / 2), int((resize_short - img_rows) / 2), int((resize_long - img_cols) / 2) + img_cols, int((resize_short - img_rows) / 2) + img_rows))

    img_1 = img_to_array(img_1)
    img_2 = img_to_array(img_2)
    img_3 = img_to_array(img_3)
    img_4 = img_to_array(img_4)
    img_5 = img_to_array(img_5)

    img_1 = scale(img_1)
    img_2 = scale(img_2)
    img_3 = scale(img_3)
    img_4 = scale(img_4)
    img_5 = scale(img_5)

    return img_1, img_2, img_3, img_4, img_5

# Make the shorter side to 256, then get the middle part

def rapid(img):
    resize_short = 128
    if img.size[0] < img.size[1]:
        resize_long = int(float(img.size[1]) / float(img.size[0]) * resize_short)
        img = scipy.misc.imresize(img, (resize_long, resize_short, channel))
        img = array_to_img(img)
        step = (resize_long - img_cols) / 2
        img = img.crop((0, step, img_cols, step + img_rows))

    if img.size[0] >= img.size[1]:
        resize_long = int(float(img.size[0]) / float(img.size[1]) * resize_short)
        img = scipy.misc.imresize(img, (resize_short, resize_long, channel))
        img = array_to_img(img)
        step = (resize_long - img_cols) / 2
        img = img.crop((step, 0, step + img_cols, img_rows))

    img = img_to_array(img)
    img = scale(img)
    return img

# Convert seconds to hours, minutes, and seconds.

def convert_time(time):
    hours = int(time/3600)
    minutes = int((time % 3600)/60)
    seconds = int(time % 60)
    return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# convert rate from number to string.

def convert_rate(rate):
    if rate == 0:
        return "Low"
    if rate == 1:
        return "High"

'''

# It is code used before.

# Transfer y to categorical

def categorical(y, real):
    y = y * 2 - 2
    c = np.zeros((real,num_classes))
    for i in range(real):
        c[i,int(y[i])] = 1
    return c

# Classify rating to different classes

def classify(rate):
    dif = rate - int(rate)
    if (dif >= 0) and (dif < 0.25):
        return int(rate)
    elif (dif >= 0.25) and (dif < 0.75):
        return int(rate) + 0.5
    elif (dif >= 0.75) and (dif < 1):
        return int(rate) + 1

# Use an extractor

try:
    img = subprocess.check_output([d.extractor_address, img_address]) # Use extractor to convert the image data to string.
except subprocess.CalledProcessError:
    continue # If the image data cannot be converted, skip it.
img = np.fromstring(img, sep=' ') # Convert the string to array.
    img = img.reshape(d.crop_rows, d.crop_cols, d.channel) # Reshape the 1D array to 3D array.

'''
