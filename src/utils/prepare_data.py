import pandas as pd
import os
from shutil import copyfile, move
import random
from augment_data import crop_image_with_cv2

PYTHON_PATH = os.environ.get('PYTHONPATH')
src_path = '%s/data/raw_train/' % PYTHON_PATH
train_path = '%s/data/train/' % PYTHON_PATH


def copy_example(filename, label):
    destination_path = train_path + label
    src_file = src_path + filename
    destination_file = destination_path + "/" + filename

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    copyfile(src_file, destination_file)
    crop_image_with_cv2(destination_path, src_file, destination_file)


def augment_example(filename, label):
    destination_path = train_path + label
    src_file = src_path + filename
    destination_file = destination_path + "/" + filename

    crop_image_with_cv2(destination_path, src_file, destination_file)


def copy_and_augment_examples():
    whale_train = pd.read_csv('%s/data/train.csv' % PYTHON_PATH)

    if not os.path.exists(train_path):
        os.mkdir(train_path)

    # copy all of the original examples
    for index, row in whale_train.iterrows():
        copy_example(row['Image'], row['Id'])

    # augment each class by cropping examples based on contours
    for index, row in whale_train.iterrows():
        augment_example(row['Image'], row['Id'])


def sample_new_whale_class():
    path = train_path + 'new_whale'
    images = os.listdir(path)
    remove_count = len(os.listdir(path)) - 34
    images_to_delete = random.sample(images, remove_count)
    for image in images_to_delete:
        os.remove(os.path.join(path, image))


def move_test_examples():
    # moving test images so that they are compatible with dataloader
    test_src_path = '%s/data/test/' % PYTHON_PATH
    test_temp_path = '%s/data/test2/' % PYTHON_PATH
    test_dest_path = '%s/data/test/test-images/' % PYTHON_PATH
    if not os.path.exists(test_dest_path):
        move(test_src_path, test_temp_path)
        os.mkdir(test_src_path)
        move(test_temp_path, test_dest_path)

copy_and_augment_examples()
move_test_examples()
sample_new_whale_class()
