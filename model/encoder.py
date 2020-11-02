import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.ops import RoIPool


class RESNET50Encoder(nn.Module):

    def __init__(self):

        super(RESNET50Encoder, self).__init__()

        convnet = resnet50(pretrained=True)
        self.num_outputs = 1024  # FIXED
        self.spatial_scale = 1/16

        self.backbone = nn.Sequential(*list(convnet.children())[:-3])  # (bz, 1024, 14, 14)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=self.spatial_scale)  #  7/224

        self.__init()

    def __init(self):

        # freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input_, scaled_bbox):
        """

        :param img_tensor: (tensor) shape (batch_size, 3, 224, 224)
        :param scaled_bbox: (list[tuple]) [(xmin, ymin, xmax, ymax), ...]
        :return: (batch_size, 1024) (batch_size, 1024)
        """

        if input_.shape[1] == 3:  # image
            feature_map = self.encode_image(input_)
        else:  # feature map
            feature_map = input_
        x = self.pooling(feature_map)
        global_feature = self.flatten(x)
        bbox_feature = self.encode_bbox(feature_map, scaled_bbox)

        return global_feature, bbox_feature

    def encode_image(self, img_tensor):
        """

        :param img_tensor: (tensor) shape (batch_size, 3, 224, 224)
        :return: (batch_size, 1024) (batch_size, 1024, 14, 14)
        """

        feature_map = self.backbone(img_tensor)

        return feature_map

    def encode_bbox(self, feature_map, scaled_bbox):
        """

        :param feature_map: (tensor) (batch_size, 1024, 14, 14)
        :param scaled_bbox: (list[tuple]) [(xmin, ymin, xmax, ymax), ...]
        :return: (batch_size, 1024)
        """

        roi = [torch.tensor([box], dtype=torch.float).to(feature_map.device) for box in scaled_bbox]  # [(1, 4), ...]

        bbox_map = self.roi_pool(feature_map, roi)
        x = self.pooling(bbox_map)
        bbox_feature = self.flatten(x)

        return bbox_feature
