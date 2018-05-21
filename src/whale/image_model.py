from torchvision.models.resnet import resnet18
from torchvision.models.alexnet import alexnet


class ImageModel:
    @staticmethod
    def built_in_models():
        # this a method, because when it is a class variable the
        # mocked models are inaccessible and it starts downloading weights
        return {
            'resnet18': resnet18,
            'alexnet': alexnet
        }

    @staticmethod
    def create_model(model_name):
        model = ImageModel.built_in_models().get(model_name, None)
        if model is None:
            raise ImageModel.ModelNotFound('model: ', model_name, ' not found')
        return model(True)

    class ModelNotFound(BaseException):
        pass
