import torch
import numpy as np
from metro.utils.geometric_layers import orthographic_projection


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        super(GAZEFROMBODY, self).__init__()
        self.bert = bert
        self.encoder1 = torch.nn.Linear(3*14,32)
        self.tanh = torch.nn.Tanh()
        self.encoder2 = torch.nn.Linear(32,3)
        self.flatten  = torch.nn.Flatten()

        self.body_mlp1 = torch.nn.Linear(args.n_frames*14*3,32)
        self.body_tanh = torch.nn.Tanh()
        self.body_mlp2 = torch.nn.Linear(32,3)


        self.metromodule = []
        for _ in range(args.n_frames):
            self.metromodule.append(bert)

        self.n_frame = args.n_frames


    def transform_head(self, pred_3d_joints):
        Nose = 13

        pred_head = pred_3d_joints[:, Nose,:]
        return pred_3d_joints - pred_head[:, None, :]

    def transform_body(self, pred_3d_joints):
        Torso = 12

        pred_torso = pred_3d_joints[:, Torso,:]
        return pred_3d_joints - pred_torso[:, None, :]


    def forward(self, image, images, smpl, mesh_sampler, is_train=False):
        self.bert.eval()

        for i in range(self.n_frame):
            self.metromodule[i].eval()

        batch_size = image.size(0)

        RSholder = 7
        LSholder = 10
        Nose = 13
        Head = 9
        Torso = 12

        pred_joints = []
        with torch.no_grad():
            for i in range(self.n_frame):
                _, tmp_joints, _, _, _, _, _, _ = self.metromodule[i](images[i], smpl, mesh_sampler)
                tmp_head_joints = self.transform_head(tmp_joints)
                pred_joints.append(tmp_head_joints)

        pred_joints = torch.stack(pred_joints, dim=3)

        reshaped_pred_joints =pred_joints.view(batch_size, -1)
        mx = self.body_mlp1(reshaped_pred_joints)
        mx = self.body_tanh(mx)
        mx = self.body_mlp2(mx)
        mdir = mx

        # metro inference
        _, pred_3d_joints, _, _, _, _, _, _ = self.bert(image, smpl, mesh_sampler)

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
            return dir#, pred_vertices
