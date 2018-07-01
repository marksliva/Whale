import os
from PIL import Image


PYTHON_PATH = os.environ.get('PYTHONPATH')
train_path = '%s/data/train/' % PYTHON_PATH

for class_path in os.listdir(train_path):
    files = os.listdir(os.path.join(train_path, class_path))
    if len(files) == 1:
        image = Image.open(os.path.join(train_path, class_path, files[0]))
        horizontally_flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        horizontally_flipped_image.save(os.path.join(train_path, class_path, 'horizontally_flipped.jpg'))
