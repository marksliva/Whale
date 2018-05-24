from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, CenterCrop, Compose


class ImageDataset:
    def __init__(self, path, batch_size, shuffle, num_workers):
        self._path = path
        self._image_folder = ImageFolder(self._path, transform=self.crop_and_to_tensor)
        self.data_loader = DataLoader(self._image_folder, batch_size, shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def crop_and_to_tensor(pil_image):
        return Compose([
            CenterCrop(200),
            ToTensor()
        ])(pil_image)

    def __len__(self):
        return self._image_folder.__len__()
