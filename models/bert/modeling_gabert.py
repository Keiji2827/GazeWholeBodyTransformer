import torch
import copy
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
        self.tanh = torch.nn.PReLU()
        self.encoder2 = torch.nn.Linear(32,3)
        self.encoder3 = torch.nn.Linear(3*14,32)
        self.encoder4 = torch.nn.Linear(32,3)
        #self.encoder3 = torch.nn.Linear(3*90,1)
        self.flatten  = torch.nn.Flatten()
        self.flatten2  = torch.nn.Flatten()

        self.metromodule = copy.deepcopy(bert)
        self.body_mlp1 = torch.nn.Linear(14*3,32)
        self.body_tanh1 = torch.nn.PReLU()
        self.body_mlp2 = torch.nn.Linear(32,32)
        self.body_tanh2 = torch.nn.PReLU()
        self.body_mlp3 = torch.nn.Linear(32,3)


    def transform_head(self, pred_3d_joints):
        Nose = 13

        pred_head = pred_3d_joints[:, Nose,:]
        return pred_3d_joints - pred_head[:, None, :]

    def transform_body(self, pred_3d_joints):
        Torso = 12

        pred_torso = pred_3d_joints[:, Torso,:]
        return pred_3d_joints - pred_torso[:, None, :]


    def forward(self, images, smpl, mesh_sampler, is_train=False, render=False):
        batch_size = images.size(0)
        self.bert.eval()
        self.metromodule.eval()
        To4joints = [ 8, 9, 13]

        RSholder = 7
        LSholder = 10
        Nose = 13
        Head = 9
        Torso = 12

        with torch.no_grad():
            _, tmp_joints, _, _, _, _, _, _ = self.metromodule(images, smpl, mesh_sampler)

        #pred_joints = torch.stack(pred_joints, dim=3)
        pred_joints = self.transform_head(tmp_joints)
        mx = self.flatten(pred_joints)
        mx = self.body_mlp1(mx)
        mx = self.body_tanh1(mx)
        mx = self.body_mlp2(mx)
        mx = self.body_tanh2(mx)
        mx = self.body_mlp3(mx)
        mdir = mx

        # metro inference
        pred_camera, pred_3d_joints, _, _, _, _, _, _ = self.bert(images, smpl, mesh_sampler)

        pred_3d_joints_gaze = self.transform_head(pred_3d_joints)

        x = self.flatten(pred_3d_joints_gaze)
        x = self.encoder1(x)
        x = self.tanh(x)
        x = self.encoder2(x)# [batch, 3]
        #dx = torch.full(x.shape, 0.01).to("cuda")
        #l2 = torch.linalg.norm(x + dx, ord=2, axis=1)
        #l2 = torch.linalg.norm(x, ord=2, axis=1)
        dir = x + mx#/l2[:,None]


        if is_train == True:
            return dir, mdir
        if is_train == False:
            return dir#, pred_vertices, pred_camera
