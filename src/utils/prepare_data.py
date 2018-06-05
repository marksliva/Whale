import pandas as pd
import os
from shutil import copyfile

PYTHON_PATH = os.environ.get('PYTHONPATH')
src_path = '%s/data/raw_train/' % PYTHON_PATH
destination_path = '%s/data/train/' % PYTHON_PATH


def copy_example(filename, label):
    label_path = destination_path + label
    src_file = src_path + filename
    destination_file = destination_path + label + "/" + filename

    if not os.path.exists(label_path):
        os.mkdir(label_path)

    copyfile(src_file, destination_file)


def copy_examples():
    whale_train = pd.read_csv('%s/data/train.csv' % PYTHON_PATH)

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    for index, row in whale_train.iterrows():
        copy_example(row['Image'], row['Id'])


copy_examples()
