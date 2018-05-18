from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader


class ImageDataset:
    def __init__(self, path, batch_size, shuffle, num_workers):
        self._path = path
        self._image_folder = ImageFolder(self._path)
        self._data_loader = DataLoader(self._image_folder, batch_size, shuffle, num_workers)

    def __len__(self):
        return self._image_folder.__len__()
