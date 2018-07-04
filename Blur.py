import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

num_row = 30
num_col = 30
overlap = 0.2

def fix_image_size(image, expected_pixels=2E5):
    ratio = math.sqrt(float(expected_pixels) / float(image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

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

def get_param(i, j, x, y):
    if i <= x:
        rowParams = x - i
    else:
        rowParams = i - x
    if j <= y:
        colParams = y - j
    else:
        colParams = j - y

    if((rowParams + colParams) != 0):
        return rowParams + colParams
    else:
        return 1

def get_score(input_image, image_row, image_col, x, y):
    score = 0
    num = 0
    center = 0
    center_blur = 0
    outside = 0
    outside_focus = 0
    score_list = np.empty((num_row, num_col), dtype="float64")

    for i in range(num_row):
        for j in range(num_col):
            begin_row = int(i * (image_row/(num_row * (1 - overlap) + 1) * (1 - overlap)))
            end_row = int(begin_row + image_row/(num_row * (1 - overlap) + 1))
            begin_col = int(j * (image_col/(num_col * (1 - overlap) + 1) * (1 - overlap)))
            end_col = int(begin_col + image_col/(num_col * (1 - overlap) + 1))

            cropped_image = input_image[begin_row : end_row, begin_col : end_col]

            if (i >= x - ((num_row / 3) + 1) / 2 and i <= x + ((num_row / 3) + 1) / 2 and j >= y - ((num_col / 3) + 1) / 2 and j <= y + ((num_col / 3) + 1) / 2):
                center += 1
                if estimate_blur(cropped_image)[1] < 100:
                    center_blur += 1

            else:
                outside += 1
                '''
                if estimate_blur(cropped_image)[1] < 2:
                    outside -= 1
                    score_list[i][j] = 0
                    break
                '''
                if estimate_blur(cropped_image)[1] > 1000:
                    outside_focus += 1

            param = 1 / float(get_param(i, j, x, y))
            score += param * estimate_blur(cropped_image)[1]
            score_list[i, j] = estimate_blur(cropped_image)[1]

            num += 1
    return score, score_list, num, center, center_blur, outside, outside_focus

def find_max(score1, score2, score3, score4):
    score = [score1, score2, score3, score4]
    max = score1
    for i in score:
        if i[0] > max[0]:
            max = i
    return max

num_train = 100000
valid = 0

label = np.empty(10, dtype="float64")

img_address = "image/1252.jpg"

input_image = cv2.imread(img_address)
input_image = fix_image_size(input_image)

image_row = input_image.shape[0]
image_col = input_image.shape[1]

max = find_max(get_score(input_image, image_row, image_col, (num_row / 3) + 1, (num_col / 3) + 1),
               get_score(input_image, image_row, image_col, (num_row / 3) + 1, num_col - (num_col / 3) + 1),
               get_score(input_image, image_row, image_col, num_row - (num_row / 3) + 1, (num_col / 3) + 1),
               get_score(input_image, image_row, image_col, num_row - (num_row / 3) + 1, num_col - (num_col / 3) + 1))
score = max[0] / max[2]

print("Score: ", score)
print(float(max[4]) / float(max[3]) * 100, "% of the inner part is blurred")
print(float(max[6]) / float(max[5]) * 100, "% of the outer part is focus")
print(1 - float(max[2]) / float(num_row * num_col)) * 100, "% is ignored"

x = np.linspace(0, 30, 30)
y = np.linspace(0, 30, 30)
X, Y = np.meshgrid(x, y)
Height = max[1]
plt.contourf(X, Y, Height, 100, alpha = 0.6)
plt.show()