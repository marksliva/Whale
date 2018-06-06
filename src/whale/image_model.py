from torchvision.models.resnet import resnet18
from torchvision.models.alexnet import alexnet
from torch import nn


class ImageModel:
    @staticmethod
    def built_in_models():
        return {
            'resnet18': resnet18,
            # haven't gotten the models below to train
            'alexnet': alexnet
        }

    @staticmethod
    def create_model(model_name):
        model = ImageModel.built_in_models().get(model_name, None)
        if model is None:
            raise ImageModel.ModelNotFound('model: ', model_name, ' not found')


        built_in_model = model(True)

        for param in built_in_model.parameters():
            param.requires_grad = False

        #softmax = nn.Softmax2d()
        new_activation = nn.Linear(in_features=512, out_features=4254, bias=True)
        built_in_model.fc = new_activation
        return built_in_model

    class ModelNotFound(BaseException):
        pass
