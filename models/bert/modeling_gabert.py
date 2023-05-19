import torch
from torch import nn
import numpy as np
from .modeling_bert import BertLayerNorm as LayerNormClass
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from metro.utils.geometric_layers import orthographic_projection


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        super(GAZEFROMBODY, self).__init__()
        self.bert = bert
        self.encoder1 = torch.nn.Linear(9,250)
        self.encoder2 = torch.nn.Linear(250,3)
        self.encoder3 = torch.nn.Linear(3*90,1)
        self.flatten  = torch.nn.Flatten()
        self.feat_mlp1 = torch.nn.Linear(2048*7*7, 64)
        self.feat_mlp2 = torch.nn.Linear(64, 3)



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
        pre_3d_joints_copy = pred_3d_joints

        pred_head = pred_3d_joints[:, 9,:]
        #pred_center = (pred_3d_joints[:, 8,:] + pred_3d_joints[:, 9,:]) / 2
        #pred_center = pred_3d_joints[:, 13,:] - pred_center
        #print(pred_3d_joints.shape)


        #feat_dir = self.flatten(image_feat_newview)
        #feat_dir = self.feat_mlp1(feat_dir)
        #feat_dir = self.feat_mlp2(feat_dir)
        #print(feat_dir.size())
        #l2 = feat_dir[:,0]**2 + feat_dir[:,1]**2 + feat_dir[:,2]**2
        #feat_dir = feat_dir/l2[:,None]

        pred_3d_joints = pred_3d_joints - pred_head[:, None, :]
        pred_3d_joints_parts = pred_3d_joints[:,[7,10,13],:]
        #print("shape of pred_3d_joints.", pred_keypoints_3d.shape) # [1, 14, 3]
        #x = pred_3d_joints.transpose(1,2)
        x = self.flatten(pred_3d_joints_parts)
        x = self.encoder1(x)
        x = self.encoder2(x)# [batch, 3]

        #x = x + feat_dir
        l2 = x[:,0]**2 + x[:,1]**2 + x[:,2]**2
        #print(l2.shape)
        x = x/l2[:,None]
        #x = x.squeeze(2)#transpose(2,1)
        #print("shape of x.", x.shape) # [1, 14, 3]



        # convert by projection : 3D joint to 2D joint
        #print("shape of pred_3d_joints.", pred_3d_joints.shape)
        pred_2d_joints = orthographic_projection(pre_3d_joints_copy, pred_camera)

        pred_head_2d = (pred_2d_joints[:, Nose,:] +  pred_2d_joints[:, Head,:])/2
        pred_head_2d =((pred_head_2d + 1) * 0.5) * 224

        #print("shape of pred_head_2d.", pred_head_2d.shape)

        if render == False:
            return x, pred_head_2d
        if render == True:
            return x, pred_head_2d, pred_vertices, pred_camera
