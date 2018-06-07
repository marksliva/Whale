from unittest import TestCase
from src.whale.image_dataset import ImageDataset
from unittest.mock import patch, MagicMock


class ImageDatasetTest(TestCase):
    def subject(self, batch_size=4, shuffle=True, num_workers=8, map_index_to_class=False):
        return ImageDataset(self._path, batch_size, shuffle, num_workers, map_index_to_class)

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

    @staticmethod
    def with_patched_folder_loader_and_transforms(block):
        with patch('src.whale.image_dataset.ImageFolder') as _patched_image_folder, patch('src.whale.image_dataset.DataLoader') as _patched_dataset_folder:
            with patch('src.whale.image_dataset.ToTensor') as patched_to_tensor, patch('src.whale.image_dataset.Resize') as patched_resize, patch('src.whale.image_dataset.Compose') as patched_compose:
                block(patched_to_tensor, patched_resize, patched_compose)

    def setUp(self):
        self._path = 'a/fake/path'
        self._described_class = ImageDataset

    def test_has_the_expected_path(self):
        def block(_patched_image_folder):
            assert self.subject()._path == self._path

        self.with_patched_image_folder(block)

    def test_creates_ImageFolder_with_the_expected_path(self):
        def block(patched_image_folder):
            assert self.subject()._path == self._path
            patched_image_folder.assert_called_with(self._path, transform=self.subject().resize_and_to_tensor)

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
            assert self.subject().data_loader == mock_data_loader

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
            ).data_loader

            assert data_loader == mock_data_loader
            patched_data_loader.assert_called_with(
                mock_image_folder,
                batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )

        self.with_patched_image_loader_and_image_folder(block)

    def test_resize_and_to_tensor_applies_a_composition_of_transforms_to_the_image(self):
        def block(patched_to_tensor, patched_resize, patched_compose):
            compose_return_value = 'return compose'
            mock_compose = MagicMock('mocked compose')
            mock_compose.return_value = compose_return_value
            patched_compose.return_value = mock_compose
            mock_resize = 'resize'
            patched_resize.return_value = mock_resize
            mock_to_tensor = 'to tensor'
            patched_to_tensor.return_value = mock_to_tensor
            pil_image = 'mock image'

            composed = self._described_class.resize_and_to_tensor(pil_image)

            patched_resize.assert_called_with((224, 224))
            patched_compose.assert_called_with([mock_resize, mock_to_tensor])
            mock_compose.assert_called_with(pil_image)
            assert composed == compose_return_value

        self.with_patched_folder_loader_and_transforms(block)

    def test_remaps_the_class_lookup_to_be_index_based_when_initialized_with_flag(self):
        def block(patched_image_folder):
            class_to_index_dictionary = {
                'foo': 0,
                'bar': 1,
                'baz': 2,
            }
            expected_index_to_class_dictionary = {
                0: 'foo',
                1: 'bar',
                2: 'baz'
            }
            mocked_image_folder = MagicMock('mocked image folder')
            mocked_image_folder.class_to_idx = class_to_index_dictionary
            patched_image_folder.return_value = mocked_image_folder
            assert self.subject(map_index_to_class=True).index_to_class_dictionary == expected_index_to_class_dictionary

        self.with_patched_image_folder(block)

    def test_does_not_remap_the_class_lookup_to_be_index_based_when_missing_flag(self):
        def block(patched_image_folder):
            mocked_image_folder = MagicMock('mocked image folder')
            mocked_image_folder.class_to_idx = {'some key': 'value'}
            patched_image_folder.return_value = mocked_image_folder
            assert self.subject().index_to_class_dictionary == {}

        self.with_patched_image_folder(block)
