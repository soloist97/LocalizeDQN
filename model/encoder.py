import torch
from torch import nn
from torchvision.models import vgg16
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

    def encode_bbox(self, feature_map, bbox, scale_factors):
        """

        :param feature_map: (tensor) (1, 512, 14, 14)
        :param bbox: (tuple) (xmin, ymin, xmax, ymax)
        :param scale_factors: (tuple) (width_factor, height_factor)
        :return: (1, 4096)
        """

        assert feature_map.shape[0] == 1, "Do not support batch input"
        assert isinstance(bbox, tuple) and len(bbox) == 4, 'invalid bbox input'
        assert isinstance(scale_factors, tuple) and len(scale_factors) == 2, 'invalid scale_factors input'

        scaled_bbox = (bbox[0] * scale_factors[0], bbox[1] * scale_factors[1],
                       bbox[2] * scale_factors[0], bbox[3] * scale_factors[1])
        roi = [torch.tensor([scaled_bbox], dtype=torch.float).to(feature_map.device)]  # [(1, 4)]

        bbox_map = self.roi_pool(feature_map, roi)
        x = self.flatten(bbox_map)
        bbox_feature = self.bbox_fc(x)

        return bbox_feature
