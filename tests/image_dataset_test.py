from unittest import TestCase
from hamcrest import *
from src.whale.image_dataset import ImageDataset
import os

class ImageDatasetTest(TestCase):
    def setUp(self):
        self._path = 'tests/fixtures'
        self._data_loader = ImageDataset(self._path)

    def test_has_the_expected_path(self):
        assert_that(self._data_loader._path, equal_to(self._path))

    def test_has_image_files(self):
        expected_examples = len(os.listdir(self._path))
        assert_that(self._data_loader.__len__(), expected_examples)


