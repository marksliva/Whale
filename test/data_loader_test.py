from unittest import TestCase
from hamcrest import *
from app.data_loader import DataLoader

class DataLoaderTest(TestCase):
    def setUp(self):
        self._path = 'not a real path'
        self._loader = DataLoader(self._path)

    def test_has_the_expected_path(self):
        assert_that(self._loader._path, equal_to(self._path))
