import os
import torch

from src.whale.image_dataset import ImageDataset
from src.whale.image_model import ImageModel


torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        model = ImageModel.create_model('resnet18')
        self._model = model.to(device)
        self._image_dataset = ImageDataset(os.getcwd() + '/../../data/train', 1, True, 2)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(list(self._model.parameters())[-2:], lr=0.00007, momentum=0.9)

    def train(self):
        temp = 0
        self._model.train()
        epochs = 10
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self._image_dataset.data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                self._optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self._model(inputs)
                    preds = torch.nn.Softmax(outputs)
                #print('predictions ', preds)
                    temp += 1
                    loss = self._criterion(outputs, labels)

                    running_loss += loss.item()

                    #print(running_loss / temp)

                    loss.backward()
                    self._optimizer.step()
            epoch_loss = running_loss / 8
            print(epoch_loss)

Trainer().train()
