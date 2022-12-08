import torch
from torch import nn
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler

class GAZEBERT(BertPreTrainedModel):
    '''
    The archtecture of a transformer encoder block we used in GAZEBERT
    '''
    def __init__(self, config):
        super(GAZEBERT, self).__init__(config)
        self.config = config
        self.bert = GAZEBERT_Encorder(config)

class GAZEBERT_Network(torch.nn.Module):

    def __init__(self, args, config, backbone, trans_encoder):
        super(GAZEBERT_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.conv_learn_tokens = torch.nn.Conv1d(48,431+14,1)
        #self.trans_encoder = trans_encoder
        self.mlp_layer1 = torch.nn.Linear(48, 1)
        self.mlp_layer2 = torch.nn.Linear(2048, 256)
        self.mlp_layer3 = torch.nn.Linear(256, 3)

        self.mlp1 = torch.nn.Linear(3,256)
        self.mlp2 = torch.nn.Linear(256,3)


    def forward(self, images,test, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        #print("batch size", batch_size)

        # extract image feature maps using a CNN backbone
        image_feat = self.backbone(images) # [32, 2048, 8 ,6]
        image_feat_newview  = image_feat.view(batch_size, 2048, -1)
        image_feat_newview2 = image_feat_newview.transpose(1,2) # [32, 48, 2048]
        img_tokens = self.conv_learn_tokens(image_feat_newview2)
        #print("size of image_ feat",image_feat_newview.size())
        
        x = self.mlp_layer1(image_feat_newview)
        x = x.transpose(1,2)
        x = self.mlp_layer2(x)
        x = self.mlp_layer3(x)
        x = x.squeeze()
        #print("size of x", x.size())
        #print("pred_gaze:",torch.Tensor(pred_gaze).size())
        
        #x = self.mlp1(test)
        #x = self.mlp2(x)

        return x

