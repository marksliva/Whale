from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize


class ImageDataset:
    def __init__(self, path, batch_size, shuffle, num_workers, map_index_to_class):
        self._path = path
        self._image_folder = ImageFolder(self._path, transform=self.resize_and_to_tensor)
        index_to_class_dictionary = self.swap_keys_values(self._image_folder.class_to_idx) if map_index_to_class else {}
        self.index_to_class_dictionary = index_to_class_dictionary
        self.filenames = self.parse_image_filenames(self._image_folder.imgs)
        self.data_loader = DataLoader(self._image_folder, batch_size, shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def resize_and_to_tensor(pil_image):
        return Compose([
            Resize((224, 224)),
            ToTensor()
        ])(pil_image)

    @staticmethod
    def parse_image_filenames(full_paths):
        return list(map(lambda path: path[0].split('/')[-1], full_paths))

    @staticmethod
    def swap_keys_values(dictionary):
        swapped_dictionary = {}
        for k in dictionary:
            v = dictionary[k]
            swapped_dictionary[v] = k
        return swapped_dictionary

    def __len__(self):
        return self._image_folder.__len__()
