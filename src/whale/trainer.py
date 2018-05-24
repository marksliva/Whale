import os
from src.whale.image_dataset import ImageDataset
from src.whale.image_model import ImageModel


class Trainer:
    def __init__(self):
        self._model = ImageModel.create_model('resnet18')
        self._image_dataset = ImageDataset(os.getcwd() + '/../../data/train', 1, True, 2)

    def train(self):
        temp = 0
        self._model.train()
        epochs = 1
        for epoch in range(epochs):
            for batch in self._image_dataset.data_loader:
                self._model(batch[0])
                temp += 1


Trainer().train()
