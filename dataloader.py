import math

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from prefetch_generator import BackgroundGenerator

from PIL import Image


class DataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Resize(object):
    """Resize the longest edge of input PIL Image to given size and keep the ratio of the image

    Args:
        max_size (int): expected max_size of the longest edge
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, max_size, interpolation=Image.BILINEAR):

        assert isinstance(max_size, int)

        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        width, height = img.size
        if width > height:
            size = (self.max_size, int(height * self.max_size / width))
        else:
            size = (int(width * self.max_size / height), self.max_size)

        return img.resize(size, self.interpolation)


class VOCLocalization(VOCDetection):

    @staticmethod
    def transform_for_tensor(max_size):

        transform = transforms.Compose([
            Resize(max_size),
            transforms.ToTensor(),  # (C, H, W) between [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    @staticmethod
    def transform_for_img(max_size):

        transform = transforms.Compose([
            Resize(max_size)
        ])

        return transform

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        N = max(batch[0][0].shape[-2:])
        padded_tensor = torch.zeros(len(batch), batch[0][0].shape[-3], N, N, dtype=batch[0][0].dtype)
        for i, b in enumerate(batch):
            C, H, W = b[0].shape
            padded_tensor[i, :C, :H, :W] = b[0]
        # stack first tensors, add rests to their correspond lists
        return (padded_tensor,) + tuple(zip(*[b[1:] for b in batch]))

    def __init__(self, *args, **kwargs):

        super(VOCLocalization, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        data_tuple = super(VOCLocalization, self).__getitem__(index)  # (image, xml-dict)

        w = float(data_tuple[1]['annotation']['size']['width'])
        h = float(data_tuple[1]['annotation']['size']['height'])

        if isinstance(data_tuple[0], torch.Tensor):
            _, hs, ws = data_tuple[0].shape  # (C, H, W)
        else:
            ws, hs = data_tuple[0].size  # (W, H)

        object_bbox_list = list()

        if isinstance(data_tuple[1]['annotation']['object'], list):
            all_objects = data_tuple[1]['annotation']['object']
        else:
            all_objects = [data_tuple[1]['annotation']['object']]

        for object in all_objects:
            bbox = [float(math.floor(float(object['bndbox']['xmin']) * ws / w)),
                    float(math.floor(float(object['bndbox']['ymin']) * hs / h)),
                    float(math.floor(float(object['bndbox']['xmax']) * ws / w)),
                    float(math.floor(float(object['bndbox']['ymax']) * hs / h))]
            object_bbox_list.append(torch.tensor(bbox, dtype=torch.float))

        return data_tuple[0], (ws, hs), object_bbox_list, index


class CombinedDataset(Dataset):

    def __init__(self, *datasets):

        self.datasets = datasets

    def __getitem__(self, index):

        i = index
        for d in self.datasets:
            if i < len(d):
                return (*d.__getitem__(i)[:-1], index)
            else:
                i -= len(d)

    def __len__(self):

        return sum(len(d) for d in self.datasets)
