import torch
#from torch import nn
#import numpy as np
#from .modeling_bert import BertLayerNorm as LayerNormClass
#from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from metro.utils.geometric_layers import orthographic_projection


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        super(GAZEFROMBODY, self).__init__()
        self.bert = bert
        self.encoder1 = torch.nn.Linear(3*14,32)
        self.encoder2 = torch.nn.Linear(32,3)
        self.encoder3 = torch.nn.Linear(3*14,32)
        self.encoder4 = torch.nn.Linear(32,3)
        #self.encoder3 = torch.nn.Linear(3*90,1)
        self.flatten  = torch.nn.Flatten()
        self.flatten2  = torch.nn.Flatten()
        #self.feat_mlp1 = torch.nn.Linear(2048*7*7, 64)
        #self.feat_mlp2 = torch.nn.Linear(64, 3)



    def forward(self, images, smpl, mesh_sampler, test=None, is_train=False, render=False):
        batch_size = images.size(0)
        self.bert.eval()
        To4joints = [ 8, 9, 13]

        RSholder = 7
        LSholder = 10
        Nose = 13
        Head = 9
        Torso = 12


        # metro inference
        pred_camera, pred_3d_joints, _, _, _, _, _, _ = self.bert(images, smpl, mesh_sampler)

        pred_head = pred_3d_joints[:, Nose,:]
        pred_torso = pred_3d_joints[:, Torso,:]
        pred_3d_joints_gaze = pred_3d_joints - pred_head[:, None, :]

        x = self.flatten(pred_3d_joints_gaze)
        x = self.encoder1(x)
        x = self.encoder2(x)# [batch, 3]
        #dx = torch.full(x.shape, 0.01).to("cuda")
        #l2 = torch.linalg.norm(x + dx, ord=2, axis=1)
        l2 = torch.linalg.norm(x, ord=2, axis=1)
        dir = x/l2[:,None]

        #pred_3d_joints_body = pred_3d_joints - pred_torso[:, None, :]
        #bx = self.flatten2(pred_3d_joints_body)
        #bx = self.encoder3(bx)
        #bx = self.encoder4(bx)# [batch, 3]
        #bdx = torch.full(bx.shape, 0.01).to("cuda")
        #bl2 = torch.linalg.norm(bx + bdx, ord=2, axis=1)
        #bl2 = torch.linalg.norm(bx, ord=2, axis=1)
        #bdir = bx#/bl2[:,None]


        # convert by projection : 3D joint to 2D joint
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

        pred_head_2d = pred_2d_joints[:, Nose,:]
        pred_head_2d =((pred_head_2d + 1) * 0.5) * 224

        pred_body_2d = pred_2d_joints[:, Torso,:]
        pred_body_2d =((pred_body_2d + 1) * 0.5) * 224

        if is_train == True:
            return dir, pred_head_2d, pred_body_2d#, bdir
        if is_train == False:
            return dir#, pred_vertices, pred_camera
