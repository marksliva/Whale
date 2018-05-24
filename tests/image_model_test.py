from unittest import TestCase
from src.whale.image_model import ImageModel
from unittest.mock import patch, MagicMock
from pytest import raises


class ImageModelTest(TestCase):
    def setUp(self):
        self._described_class = ImageModel

    def subject(self, model_name):
        return self._described_class.create_model(model_name)

    @staticmethod
    def with_patched_resnet_model(block):
        with patch('src.whale.image_model.resnet18') as patched_resnet18_model, patch('src.whale.image_model.nn.Linear') as patched_linear_layer:
            block(patched_resnet18_model, patched_linear_layer)

    @staticmethod
    def with_patched_alexnet_model(block):
        with patch('src.whale.image_model.alexnet') as patched_alexnet_model, patch('src.whale.image_model.nn.Linear') as patched_linear_layer:
            block(patched_alexnet_model, patched_linear_layer)

    def test_creating_a_built_in_resnet_model_creates_a_pretrained_model(self):
        def block(patched_model, patched_linear_layer):
            expected_linear_layer = 'mock linear layer'
            patched_linear_layer.return_value = expected_linear_layer
            expected_model = MagicMock('resnet 18 model')
            patched_model.return_value = expected_model

            actual_model = self.subject('resnet18')

            patched_model.assert_called_with(True)
            assert actual_model == expected_model

            patched_linear_layer.assert_called_with(in_features=512, out_features=4254, bias=True)
            assert expected_model.fc == expected_linear_layer

        self.with_patched_resnet_model(block)

    def test_creating_a_built_in_alexnet_model_creates_a_pretrained_model(self):
        def block(patched_model, patched_linear_layer):
            expected_linear_layer = 'mock linear layer'
            patched_linear_layer.return_value = expected_linear_layer
            expected_model = MagicMock('alexnet model')
            patched_model.return_value = expected_model

            actual_model = self.subject('alexnet')

            patched_model.assert_called_with(True)
            assert actual_model == expected_model

            patched_linear_layer.assert_called_with(in_features=512, out_features=4254, bias=True)
            assert expected_model.fc == expected_linear_layer

        self.with_patched_alexnet_model(block)

    def test_creating_an_unsupported_model_type_raises_an_error(self):
        with raises(ImageModel.ModelNotFound):
            self.subject('George Washington')
