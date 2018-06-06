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
        # self._train_dataset = ImageDataset(os.getcwd() + '/../../data/train', 1, True, 2)
        # self._test_dataset = ImageDataset(os.getcwd() + '/../../data/test', 1, True, 2)
        self._train_dataset = ImageDataset(os.getcwd() + '/../../tests/fixtures/image_dataset_train', 1, True, 2)
        self._test_dataset = ImageDataset(os.getcwd() + '/../../tests/fixtures/image_dataset_train', 1, True, 2)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(list(self._model.parameters())[-2:], lr=0.00007, momentum=0.9)

    def train(self):
        self._model.train()
        epochs = 50
        for epoch in range(epochs):
            running_loss = 0.0

            # train phase
            for inputs, labels in self._train_dataset.data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                self._optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # todo: figure out how to get this working for a backwards pass
                    # softmax = torch.nn.Softmax()
                    # outputs = softmax(self._model(inputs))
                    outputs = self._model(inputs)

                    loss = self._criterion(outputs, labels)

                    running_loss += loss.item()

                    loss.backward()
                    self._optimizer.step()
            epoch_loss = running_loss / self._train_dataset.__len__()
            print('loss on the training set: ', epoch_loss)

            running_loss = 0.0

            # test phase
            for inputs, labels in self._test_dataset.data_loader:
                self._model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                running_loss += loss

            print('loss on the test set: ', running_loss / self._test_dataset.__len__())

Trainer().train()
