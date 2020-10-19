import torch
from torch.utils.data import DataLoader, Dataset
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (C, H, W) between [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        # stack first tensors, add rests to their correspond lists
        return (torch.stack([b[0] for b in batch], dim=0),) + tuple(zip(*[b[1:] for b in batch]))

    def __init__(self, *args, **kwargs):

        super(VOCLocalization, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        data_tuple = super(VOCLocalization, self).__getitem__(index)  # (image, xml-dict)

        original_shape = (
            float(data_tuple[1]['annotation']['size']['width']),
            float(data_tuple[1]['annotation']['size']['height']),
        )

        object_bbox_list = list()

        if isinstance(data_tuple[1]['annotation']['object'], list):
            all_objects = data_tuple[1]['annotation']['object']
        else:
            all_objects = [data_tuple[1]['annotation']['object']]

        for object in all_objects:
            bbox = [float(object['bndbox']['xmin']), float(object['bndbox']['ymin']),
                    float(object['bndbox']['xmax']), float(object['bndbox']['ymax'])]
            object_bbox_list.append(torch.tensor(bbox, dtype=torch.float))

        return data_tuple[0], original_shape, object_bbox_list, index


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
