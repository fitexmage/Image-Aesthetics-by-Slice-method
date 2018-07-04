import numpy as np
import math
import cv2
from sklearn import preprocessing

# Version

version = "0.0.1"

# Data for loading images

num_rows, num_cols, channel = 64, 64, 3 # The size of images after resizing and cropping.
overlap = 0.4
num_train = 1000 # Number of training data
num_test = 500 # Number of testing data
least_rate = 10 # The least times of rating
large_border = 2 # The largest border of images
display_step = 10 # The display step for loading the data

# Address

text_address = "data/rbf.txt"
image_address_train = "/backups4/fuf111/DATASETs/photonet/original/"
image_address_test = "/backups4/fuf111/DATASETs/photonet/original/"
model_address = "data/model.json"
parameters_address = "data/model.h5"
result_address = "data/result.txt"
error_address = "data/error.txt"

# Data for model

batch_size = 16 # Batch size
num_classes = 4 # Number of classes
epochs = 10 # Epochs

# Method

# Transfer y to categorical

def categorical(label):
    len = len(label)
    c = np.zeros((len, num_classes))
    for i in range(len):
        c[i,int(label[i])] = 1
    return c

# Classify rating to different classes

def classify(rate):

    if rate < 4.5:
        return 0
    elif 4.5 <= rate < 5:
        return 1
    elif 5 <= rate < 5.5:
        return 2
    else:
        return 3

# Convert seconds to hours, minutes, and seconds.

def convert_time(time):
    hours = int(time/3600)
    minutes = int((time % 3600)/60)
    seconds = int(time % 60)
    return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Fix the image size to fixed area

def fix_image_size(image, expected_pixels=2E5):
    ratio = math.sqrt(float(expected_pixels) / float(image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

# Estimate the blur

def estimate_blur(image, threshold=100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)

def pretty_blur_map(blur_map, sigma=5):
    abs_image = np.log(np.abs(blur_map).astype(np.float32))
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)

# Get param based on distance to the point
# i and j are for the cropped image, x and y are for the fixed point

def get_param(i, j, x, y):
    if i < x - num_rows / 5:
        rowParams = x - num_rows / 5 - i
    elif i > x + num_rows / 5:
        rowParams = i - x - num_rows / 5
    else:
        rowParams = 0

    if j < y - num_cols / 5:
        colParams = y - num_cols / 5 - j
    elif j > y + num_cols / 5:
        colParams = j - y - num_cols / 5
    else:
        colParams = 0

    if ((rowParams + colParams) != 0):
        return rowParams + colParams
    else:
        return 1

# Get score list

def get_score_list(input_image):
    score_list = np.empty((num_rows, num_cols), dtype="float64")
    image_row = input_image.shape[0]
    image_col = input_image.shape[1]
    total_blur = 0

    for i in range(num_rows):
        for j in range(num_cols):
            begin_row = int(i * (image_row / (num_rows * (1 - overlap) + 1) * (1 - overlap)))
            end_row = int(begin_row + image_row / (num_rows * (1 - overlap) + 1))
            begin_col = int(j * (image_col / (num_cols * (1 - overlap) + 1) * (1 - overlap)))
            end_col = int(begin_col + image_col / (num_cols * (1 - overlap) + 1))

            cropped_image = input_image[begin_row: end_row, begin_col: end_col]

            score_list[i, j] = estimate_blur(cropped_image)[1]
            total_blur += estimate_blur(cropped_image)[1]

    for i in range(num_rows):
        for j in range(num_cols):
            score_list[i, j] = score_list[i, j] / total_blur

    '''
    my_scaler = preprocessing.StandardScaler()
    my_scaler.fit(score_list)
    my_scaler.transform(score_list)
    '''

    return score_list

# Get score for a fixed point

def get_score(input_image, x, y):
    score = 0
    num = 0
    center = 0
    center_blur = 0
    outside = 0
    outside_focus = 0
    score_list = np.empty((num_rows, num_cols), dtype="float64")
    image_row = input_image.shape[0]
    image_col = input_image.shape[1]

    for i in range(num_rows):
        for j in range(num_cols):
            begin_row = int(i * (image_row / (num_rows * (1 - overlap) + 1) * (1 - overlap)))
            end_row = int(begin_row + image_row / (num_rows * (1 - overlap) + 1))
            begin_col = int(j * (image_col / (num_cols * (1 - overlap) + 1) * (1 - overlap)))
            end_col = int(begin_col + image_col / (num_cols * (1 - overlap) + 1))

            cropped_image = input_image[begin_row: end_row, begin_col: end_col]

            if (x - ((num_rows / 3) + 1) / 2 <= i <= x + ((num_rows / 3) + 1) / 2
                and y - ((num_cols / 3) + 1) / 2 <= j <= y + ((num_cols / 3) + 1) / 2):
                center += 1
                if estimate_blur(cropped_image)[1] < 100:
                    center_blur += 1

            else:
                outside += 1
                if estimate_blur(cropped_image)[1] > 1000:
                    outside_focus += 1

            param = 1 / float(get_param(i, j, x, y))
            score += param * estimate_blur(cropped_image)[1]
            score_list[i, j] = estimate_blur(cropped_image)[1]

            num += 1
    return score, score_list, num, center, center_blur, outside, outside_focus

# Find the max score

def find_max(score1, score2, score3, score4):
    score = [score1, score2, score3, score4]
    max = score1
    for i in score:
        if i[0] > max[0]:
            max = i
    return max

# Get color score

def get_color(image):
    image_row = image.shape[0]
    image_col = image.shape[1]
    color = np.zeros(3, dtype="float64")

    for i in range(image_row):
        for j in range(image_col):
            color[0] += image[i, j, 0]
            color[1] += image[i, j, 1]
            color[2] += image[i, j, 2]

    color[0] /= i * j
    color[1] /= i * j
    color[2] /= i * j
    return color

def get_color_diff(color1, color2):
    return math.sqrt(pow((color1[0] - color2[0]), 2)+ pow((color1[1] - color2[1]), 2) + pow((color1[2] - color2[2]), 2))

def get_color_list(input_image):
    color_list = np.empty((num_rows, num_cols, 3), dtype="float64")
    color_diff_list = np.empty((num_rows, num_cols), dtype="float64")
    image_row = input_image.shape[0]
    image_col = input_image.shape[1]

    for i in range(num_rows):
        for j in range(num_cols):
            begin_row = int(i * (image_row / (num_rows * (1 - overlap) + 1) * (1 - overlap)))
            end_row = int(begin_row + image_row / (num_rows * (1 - overlap) + 1))
            begin_col = int(j * (image_col / (num_cols * (1 - overlap) + 1) * (1 - overlap)))
            end_col = int(begin_col + image_col / (num_cols * (1 - overlap) + 1))

            cropped_image = input_image[begin_row: end_row, begin_col: end_col]
            color_list[i, j, :] = get_color(cropped_image)
    for i in range(num_rows):
        for j in range(num_cols):
            count = 0
            if i != 0:
                count += 1
                color_diff_list[i, j] += get_color_diff(color_list[i, j], color_list[i - 1, j])
            if i != num_rows - 1:
                count += 1
                color_diff_list[i, j] += get_color_diff(color_list[i, j], color_list[i + 1, j])
            if j != 0:
                count += 1
                color_diff_list[i, j] += get_color_diff(color_list[i, j], color_list[i, j - 1])
            if j != num_cols - 1:
                count += 1
                color_diff_list[i, j] += get_color_diff(color_list[i, j], color_list[i, j + 1])

            color_diff_list[i, j] = color_diff_list[i, j]/count

    '''
    my_scaler = preprocessing.StandardScaler()
    my_scaler.fit(color_diff_list)
    my_scaler.transform(color_diff_list)
    '''

    return color_diff_list

def get_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_row = image.shape[0]
    image_col = image.shape[1]
    hsv = np.zeros(3, dtype="float64")

    for i in range(image_row):
        for j in range(image_col):
            hsv[0] += image[i, j, 0]
            hsv[1] += image[i, j, 1]
            hsv[2] += image[i, j, 2]

    hsv[0] /= i * j
    hsv[1] /= i * j
    hsv[2] /= i * j
    return hsv

def get_hsv_list(input_image):
    hsv_list = np.empty((num_rows, num_cols, 3), dtype="float64")
    image_row = input_image.shape[0]
    image_col = input_image.shape[1]

    for i in range(num_rows):
        for j in range(num_cols):
            begin_row = int(i * (image_row / (num_rows * (1 - overlap) + 1) * (1 - overlap)))
            end_row = int(begin_row + image_row / (num_rows * (1 - overlap) + 1))
            begin_col = int(j * (image_col / (num_cols * (1 - overlap) + 1) * (1 - overlap)))
            end_col = int(begin_col + image_col / (num_cols * (1 - overlap) + 1))

            cropped_image = input_image[begin_row: end_row, begin_col: end_col]
            hsv_list[i, j, :] = get_hsv(cropped_image)

    return hsv_list
