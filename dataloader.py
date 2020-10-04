import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from prefetch_generator import BackgroundGenerator


class DataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class VOCLocalization(VOCDetection):

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.ToTensor(),  # (C, H, W) between [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self, *args, **kwargs):

        super(VOCLocalization, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        data_tuple = super(VOCLocalization, self).__getitem__(index)  # (image, xml-dict)

        object_bbox_list = list()
        for object in data_tuple[1]['annotation']['object']:
            bbox = [float(object['bndbox']['xmin']), float(object['bndbox']['ymin']),
                    float(object['bndbox']['xmax']), float(object['bndbox']['ymax'])]
            object_bbox_list.append(torch.tensor(bbox, dtype=torch.float))

        return (data_tuple[0], object_bbox_list)
