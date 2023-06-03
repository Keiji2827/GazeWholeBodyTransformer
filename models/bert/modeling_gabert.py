import torch
from torch import nn
import numpy as np
from .modeling_bert import BertLayerNorm as LayerNormClass
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        super(GAZEFROMBODY, self).__init__()
        self.bert = bert
        self.encoder1 = torch.nn.Linear(9,250)
        self.encoder2 = torch.nn.Linear(250,3)
        #self.encoder3 = torch.nn.Linear(3*90,1)
        self.flatten  = torch.nn.Flatten()
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


        # metro inference
        pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, hidden_states, att, image_feat_newview = self.bert(images, smpl, mesh_sampler)
        #print("shape of pred_3d_joints.", pred_3d_joints.shape) # [1, 14, 3]

        pred_head = pred_3d_joints[:, 9,:]
 

        pred_3d_joints = pred_3d_joints - pred_head[:, None, :]
        pred_3d_joints = pred_3d_joints[:,[7,10,13],:]
        #print("shape of pred_3d_joints.", pred_keypoints_3d.shape) # [1, 14, 3]
        #x = pred_3d_joints.transpose(1,2)
        x = self.flatten(pred_3d_joints)
        x = self.encoder1(x)
        x = self.encoder2(x)# [batch, 3]
        l2 = torch.linalg.norm(x, ord=2, axis=1)
        dir = x/l2[:,None]

        if render == False:
            return dir
        if render == True:
            return dir, pred_vertices, pred_camera
