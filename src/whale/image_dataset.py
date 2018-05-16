from torchvision.datasets.folder import ImageFolder


class ImageDataset:
    def __init__(self, path):
        self._path = path
        self._image_folder = ImageFolder(path)

    def __len__(self):
        return self._image_folder.__len__()
