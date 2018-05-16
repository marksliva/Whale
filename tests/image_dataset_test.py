from unittest import TestCase
from hamcrest import *
from src.whale.image_dataset import ImageDataset
from unittest.mock import patch, MagicMock


class ImageDatasetTest(TestCase):
    def subject(self):
        return ImageDataset(self._path)

    @staticmethod
    def with_patched_image_folder(block):
        with patch('src.whale.image_dataset.ImageFolder') as patched_image_folder:
            block(patched_image_folder)

    def setUp(self):
        self._path = 'a/fake/path'

    def test_has_the_expected_path(self):
        def block(patched_image_folder):
            mock_image_folder = 'fake image folder'
            patched_image_folder.return_value = mock_image_folder
            assert_that(self.subject()._path, equal_to(self._path))

        self.with_patched_image_folder(block)

    def test_creates_an_ImageFolder(self):
        def block(patched_image_folder):
                mock_image_folder = 'fake image folder'
                patched_image_folder.return_value = mock_image_folder
                assert_that(self.subject()._image_folder, equal_to(mock_image_folder))

        self.with_patched_image_folder(block)

    def test_the_ImageFolder_returns_length_based_on_the_image_folder(self):
        def block(patched_image_folder):
            mock_image_folder = MagicMock('a mocked image folder')
            length = 42
            mock_image_folder.__len__ .return_value = length
            patched_image_folder.return_value = mock_image_folder
            assert_that(self.subject().__len__(), equal_to(length))

        self.with_patched_image_folder(block)

    # I prefer the above style, because it will be easier to refactor later
    # Also, in more complex cases where multiple things are patched, I really hate how
    # the multiple patch decorator variables end up in a reversed ordering:
    # http://www.voidspace.org.uk/python/mock/patch.html?highlight=stack#nesting-patch-decorators
    # This is the version I prefer to use:
    # https://stackoverflow.com/a/30799104
    #
    # The following test is the decorator version that I refactored away from:
    # @patch('src.whale.image_dataset.ImageFolder')
    # def test_the_ImageFolder_returns_length_based_on_the_image_folder(self, image_folder):
    #     mock_image_folder = MagicMock('a mocked image folder')
    #     mock_image_folder.__len__ .return_value = 42
    #     image_folder.return_value = mock_image_folder
    #     assert_that(self.subject().__len__(), equal_to(42))
