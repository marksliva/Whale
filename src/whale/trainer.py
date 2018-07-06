import copy
import time
import torch
import numpy as np
import csv

from src.whale.image_dataset import ImageDataset
from src.whale.image_model import ImageModel


torch.manual_seed(42)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


class Trainer:
    def __init__(self):
        model = ImageModel.create_model('resnet18')
        self._model = model.to(device)

        # todo: segment out a validation dataset from the training set
        self._train_dataset = ImageDataset('data/train', 32, True, 2, True)
        self._validation_dataset = ImageDataset('data/train', 32, False, 2, False)
        self._test_dataset = ImageDataset('data/test', 1, False, 2, False)
        # self._train_dataset = ImageDataset('tests/fixtures/image_dataset_train', 1, True, 2, True)
        # self._validation_dataset = ImageDataset('tests/fixtures/image_dataset_train', 1, False, 2, False)
        # self._test_dataset = ImageDataset('tests/fixtures/image_dataset_train', 1, False, 2, False)

        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(list(self._model.parameters())[-2:], lr=0.00007, momentum=0.9)

    def train(self):
        lowest_loss = 999999999.9
        best_model_weights = None
        self._model.train()
        epochs = 1
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

            # validation phase
            for inputs, labels in self._validation_dataset.data_loader:
                self._model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                running_loss += loss

            print('loss on the validation set: ', running_loss / self._validation_dataset.__len__())
            if running_loss < lowest_loss:
                # todo: when we segment out a validation set (after data augmentation in image_datatset),
                # we would want to update this part to use the accuracy instead. (https://stackoverflow.com/a/46800337)
                print('saving best model weights with loss: %s at epoch: %s' % (running_loss, epoch))
                lowest_loss = running_loss
                best_model_weights = copy.deepcopy(self._model.state_dict())

        # write predictions to a csv file
        with(open('predictions_%s.csv' % time.strftime('%y-%m-%d_%H:%M:%S'), 'w')) as csv_file:
            filename_index = 0
            predictions_csv = csv.writer(csv_file, delimiter=',')
            predictions_csv.writerow(['Image', 'Id'])
            self._model.load_state_dict(best_model_weights)
            weights_path = 'model_weights_%s' % time.strftime('%y-%m-%d_%H:%M:%S')
            torch.save(self._model.state_dict(), weights_path)
            self._model.eval()
            for inputs, _labels in self._test_dataset.data_loader:
                inputs = inputs.to(device)
                outputs = self._model(inputs)
                for output in outputs:
                    if use_gpu:
                        index = np.argmax(output.detach().cpu().numpy())
                    else:
                        index = np.argmax(output.detach().numpy())
                    filename = self._test_dataset.filenames[filename_index]
                    filename_index += 1
                    label = self._train_dataset.index_to_class_dictionary.get(index) or 'fail-whale'
                    predictions_csv.writerow([filename, label])


Trainer().train()
