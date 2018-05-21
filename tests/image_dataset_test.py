from unittest import TestCase
from src.whale.image_dataset import ImageDataset
from unittest.mock import patch, MagicMock


class ImageDatasetTest(TestCase):
    def subject(self, batch_size = 4, shuffle = True, num_workers = 8):
        return ImageDataset(self._path, batch_size, shuffle, num_workers)

    @staticmethod
    def with_patched_image_loader_and_image_folder(block):
        with patch('src.whale.image_dataset.ImageFolder') as patched_image_folder:
            with patch('src.whale.image_dataset.DataLoader') as patched_data_loader:
                block(patched_image_folder, patched_data_loader)

    @staticmethod
    def with_patched_image_folder(block):
        with patch('src.whale.image_dataset.ImageFolder') as patched_image_folder:
            with patch('src.whale.image_dataset.DataLoader') as _patched_data_loader:
                block(patched_image_folder)

    def setUp(self):
        self._path = 'a/fake/path'

    def test_has_the_expected_path(self):
        def block(_patched_image_folder):
            assert self.subject()._path == self._path

        self.with_patched_image_folder(block)

    def test_creates_ImageFolder_with_the_expected_path(self):
        def block(patched_image_folder):
            assert self.subject()._path == self._path
            patched_image_folder.assert_called_with(self._path)

        self.with_patched_image_folder(block)

    def test_creates_an_ImageFolder(self):
        def block(patched_image_folder):
            mock_image_folder = 'fake image folder'
            patched_image_folder.return_value = mock_image_folder
            assert self.subject()._image_folder == mock_image_folder

        self.with_patched_image_folder(block)

    def test_the_ImageFolder_returns_length_based_on_the_image_folder(self):
        def block(patched_image_folder):
            mock_image_folder = MagicMock('a mocked image folder')
            length = 42
            mock_image_folder.__len__ .return_value = length
            patched_image_folder.return_value = mock_image_folder
            assert self.subject().__len__() == length

        self.with_patched_image_folder(block)

    def test_it_creates_a_DataLoader(self):
        def block(_patched_image_folder, patched_data_loader):
            mock_data_loader = 'fake data loader'
            patched_data_loader.return_value = mock_data_loader
            assert self.subject()._data_loader == mock_data_loader

        self.with_patched_image_loader_and_image_folder(block)

    def test_it_creates_a_DataLoader_from_the_ImageFolder(self):
        def block(patched_image_folder, patched_data_loader):
            batch_size = 123
            shuffle = False
            num_workers = 50000
            mock_image_folder = 'fake image folder'
            patched_image_folder.return_value = mock_image_folder
            mock_data_loader = 'fake data loader'
            patched_data_loader.return_value = mock_data_loader
            data_loader = self.subject(
                batch_size,
                shuffle,
                num_workers
            )._data_loader

            assert data_loader == mock_data_loader
            patched_data_loader.assert_called_with(
                mock_image_folder,
                batch_size,
                shuffle,
                num_workers
            )

        self.with_patched_image_loader_and_image_folder(block)
