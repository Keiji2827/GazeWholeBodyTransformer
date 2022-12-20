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
        self.encoder3 = torch.nn.Linear(3*90,1)
        self.flatten  = torch.nn.Flatten()
        self.encoder13 = torch.nn.Linear(3,250)
        self.encoder31 = torch.nn.Linear(250,3)

        self.test1 = torch.nn.Linear(3,250)
        self.test2 = torch.nn.Linear(250,3)



    def forward(self, images, smpl, mesh_sampler, test,is_train=False):
        batch_size = images.size(0)
        self.bert.eval()
        To4joints = [ 8, 9, 13]

        # metro inference
        pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, hidden_states, att = self.bert(images, smpl, mesh_sampler)
        #print("shape of pred_3d_joints.", pred_3d_joints.shape) # [1, 14, 3]

        pred_head = pred_3d_joints[:, 9,:]
        pred_center = (pred_3d_joints[:, 8,:] + pred_3d_joints[:, 9,:]) / 2
        pred_center = pred_3d_joints[:, 13,:] - pred_center
        #print(pred_3d_joints.shape)
        #print(test)
        #print("--")
        #pred_center = pred_center.squeeze()
        #x = self.encoder13(pred_center)
        #x = self.encoder31(x)
        #return x


        #print("pred_center:",pred_center)
        pred_3d_joints = pred_3d_joints - pred_head[:, None, :]
        pred_3d_joints = pred_3d_joints[:,[7,10,13],:]
        #print("shape of pred_3d_joints.", pred_keypoints_3d.shape) # [1, 14, 3]
        #x = pred_3d_joints.transpose(1,2)
        x = self.flatten(pred_3d_joints)
        x = self.encoder1(x)
        x = self.encoder2(x)# [batch, 3]
        l2 = x[:,0]**2 + x[:,1]**2 + x[:,2]**2
        #print(l2.shape)
        x = x/l2[:,None]
        #x = x.squeeze(2)#transpose(2,1)
        #print("shape of x.", x.shape) # [1, 14, 3]
        return x

        y = self.test1(test)
        y = self.test2(y)
        return y

