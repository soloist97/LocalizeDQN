import torch
from torch import nn
from torchvision.models import vgg16, resnet50
from torchvision.ops import RoIPool


class VGG16Encoder(nn.Module):

    def __init__(self):

        super(VGG16Encoder, self).__init__()

        convnet = vgg16(pretrained=True)

        self.conv5_3 = convnet.features[:30]  # after relu
        self.pooling = nn.Sequential(
            convnet.features[30],
            convnet.avgpool,
        )
        self.flatten = nn.Flatten(start_dim=1)

        self.global_fc = convnet.classifier[:2]  # fc6: after relu before dropout
        self.bbox_fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU()
        )

        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)  #  14/224

        self.__init()

    def __init(self):

        # freeze
        for param in self.conv5_3.parameters():
            param.requires_grad = False

        # init
        self.bbox_fc.load_state_dict(self.global_fc.state_dict())

    def forward(self, img_tensor):
        """

        :param img_tensor: (tensor) shape (batch_size, 3, 224, 224)
        :return: (batch_size, 4096) (batch_size, 512, 14, 14)
        """

        feature_map = self.conv5_3(img_tensor)
        x = self.pooling(feature_map)
        x = self.flatten(x)
        global_feature = self.global_fc(x)

        return global_feature, feature_map

    def encode_bbox(self, feature_map, scaled_bbox):
        """

        :param feature_map: (tensor) (batch_size, 512, 14, 14)
        :param scaled_bbox: (list[tuple]) [(xmin, ymin, xmax, ymax), ...]
        :return: (batch_size, 4096)
        """

        roi = [torch.tensor([box], dtype=torch.float).to(feature_map.device) for box in scaled_bbox]  # [(1, 4), ...]

        bbox_map = self.roi_pool(feature_map, roi)
        x = self.flatten(bbox_map)
        bbox_feature = self.bbox_fc(x)

        return bbox_feature


class RESNET50Encoder(nn.Module):

    def __init__(self):

        super(RESNET50Encoder, self).__init__()

        convnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(convnet.children())[:-3])  # (bz, 1024, 14, 14)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)  #  7/224

        self.__init()

    def __init(self):

        # freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, img_tensor):
        """

        :param img_tensor: (tensor) shape (batch_size, 3, 224, 224)
        :return: (batch_size, 1024) (batch_size, 1024, 14, 14)
        """

        feature_map = self.backbone(img_tensor)
        x = self.pooling(feature_map)
        global_feature = self.flatten(x)

        return global_feature, feature_map

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
