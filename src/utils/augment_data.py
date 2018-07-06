import os
import cv2
import numpy as np
from PIL import Image


def horizontal_flip():
    PYTHON_PATH = os.environ.get('PYTHONPATH')
    train_path = '%s/data/train/' % PYTHON_PATH

    for class_path in os.listdir(train_path):
        files = os.listdir(os.path.join(train_path, class_path))
        if len(files) == 1:
            image = Image.open(os.path.join(train_path, class_path, files[0]))
            horizontally_flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            horizontally_flipped_image.save(os.path.join(train_path, class_path, 'horizontally_flipped.jpg'))


def crop_image_with_cv2(src_file, destination_file):
    image = cv2.imread(src_file)
    # transform colorspace
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    image_y[:, :] = image_yuv[:, :, 0]
    # Blur image to filter out high frequency noise
    image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)
    # Edge
    edges = cv2.Canny(image_blurred, 100, 300, apertureSize=3)
    # Find extreme outer countors
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Crop image
    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            idx += 1
            new_img = image[y:y + h, x:x + w]
            cv2.imwrite(destination_file + str(idx) + '.png', new_img)
