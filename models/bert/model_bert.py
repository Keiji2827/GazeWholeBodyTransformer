import torch
from torch import nn


class GAZEBERT_Network(torch.nn.Module):

    def __init__(self, args, backbone):
        super(GAZEBERT_Network, self).__init__()
        #self.config = config
        #self.config.device = args.device
        self.backbone = backbone
        #self.trans_encoder = trans_encoder

    def forward(self, images):
        batch_size = images.size()

        # extract image feature maps using a CNN backbone
        image_feat = self.backbone(images)
        #image_feat_newview = image_feat.view(batch_size, 2048, -1)
        #image_feat_newview = image_feat_newview.transpose(1,2)


        return image_feat

