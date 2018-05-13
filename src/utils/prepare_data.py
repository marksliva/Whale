import pandas as pd
import os
from shutil import copyfile

src_path = 'data/raw_train/'
destination_path = 'data/train/'

whale_train = pd.read_csv("data/train.csv")


def copy_example(filename, label):
    label_path = destination_path + label
    src_file = src_path + filename
    destination_file = destination_path + label + "/" + filename

    if not os.path.exists(label_path):
        os.mkdir(label_path)

    copyfile(src_file, destination_file)


for index, row in whale_train.iterrows():
    copy_example(row['Image'], row['Id'])
