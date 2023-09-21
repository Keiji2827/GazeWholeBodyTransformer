import os
import pickle
import cv2
import time
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, ImageOps
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class GazeSeqDataset(Dataset):
    def __init__(self, video_path, human=False, flip=False):
        self.video_path = video_path
        self.n_frames = 7
        self.flip = flip

        # load annotation
        with open(os.path.join(video_path, 'annotation.pickle'), "rb") as f:
            anno_data = pickle.load(f)


        #print(anno_data['index'][0])

        if human:

            # load index
            if os.path.exists(os.path.join(video_path, 'index.npy')):
                with open(os.path.join(video_path, 'index.npy'), "rb") as f:
                    index_data = np.load(f)
            else:
                index_data = []
                #print(index_data)

            self.bodys = []
            self.heads = []
            self.gazes = []
            self.R_cam = anno_data["R_cam"]
            self.t_cam = anno_data["t_cam"]
            self.body_pos = []
            self.head_pos = []
            self.org_index = anno_data['index']
            self.img_index = []
            self.keypoints = []
            self.head_bb = np.empty((0, 4), dtype=np.float32)
            self.body_bb = np.empty((0, 4), dtype=np.float32)

            index_num = 0
            for i in range(len(self.org_index)):

                if index_num >= len(index_data):
                    break

                if index_data[index_num] != self.org_index[i]:
                    index_num = index_num
                    continue

                self.bodys.append(anno_data["bodys"][i])
                self.heads.append(anno_data["heads"][i])
                self.gazes.append(anno_data["gazes"][i])
                self.body_pos.append(anno_data["body_pos"][i])
                self.head_pos.append(anno_data["head_pos"][i])
                self.img_index.append(index_data[index_num])
                self.keypoints.append(anno_data['keypoins'][i])
                self.head_bb = np.vstack((self.head_bb, anno_data['head_bb'][i]))
                self.body_bb = np.vstack((self.body_bb, anno_data['body_bb'][i]))
                
                index_num = index_num + 1

        else:
            self.bodys = anno_data["bodys"]
            self.heads = anno_data["heads"]
            self.gazes = anno_data["gazes"]
            self.R_cam = anno_data["R_cam"]
            self.t_cam = anno_data["t_cam"]
            self.body_pos = anno_data["body_pos"]
            self.head_pos = anno_data["head_pos"]
            self.img_index = anno_data['index']
            self.keypoints = anno_data['keypoins']
            self.head_bb = np.vstack(anno_data['head_bb']).astype(np.float32)
            self.body_bb = np.vstack(anno_data['body_bb']).astype(np.float32)
                

        # abort if no data
        if len(self.gazes) < 1:
            self.valid_index = []
            return

        # extract successive frames
        self.valid_index = []
        for i in range(0, len(self.img_index) - self.n_frames):
            if self.img_index[i] == self.img_index[i] and i < len(self.gazes):
                self.valid_index.append(i)
        self.valid_index = np.array(self.valid_index)



        # Head boundig box changed to relative to chest
        #self.head_bb = np.vstack(anno_data['head_bb']).astype(np.float32)
        #self.body_bb = np.vstack(anno_data['body_bb']).astype(np.float32)
        self.head_bb = self.head_bb.astype(np.float32)
        self.body_bb = self.body_bb.astype(np.float32)

        self.head_bb[:, 0] -= self.body_bb[:, 0]
        self.head_bb[:, 1] -= self.body_bb[:, 1]
        self.head_bb[:, [0, 2]] /= self.body_bb[:, 2][:, None]
        self.head_bb[:, [1, 3]] /= self.body_bb[:, 3][:, None]


        #self.body_bb[:, 0] -= self.head_bb[:, 0]
        #self.body_bb[:, 1] -= self.head_bb[:, 1]
        self.body_bb[:, [0, 2]] /= 1920#self.body_bb[:, 2][:, None]
        self.body_bb[:, [1, 3]] /= 1920#self.body_bb[:, 3][:, None]

        # image transform for body image
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} >= len {len(self)}")

        idx = self.valid_index[idx]
        img_path = os.path.join(self.video_path, f"{self.img_index[idx]:06}.jpg")
        #print(img_path)
        img = Image.open(img_path)
        if self.flip == True:
            img = ImageOps.mirror(img)
        img_ = transform(img)

        # create mask of head bounding box
        head_bb_int = self.head_bb[idx].copy()
        head_bb_int[ [0, 2]] *= img_.shape[2]
        head_bb_int[ [1, 3]] *= img_.shape[1]
        head_bb_int[ 2] += head_bb_int[ 0]
        head_bb_int[ 3] += head_bb_int[ 1]
        head_bb_int = head_bb_int.astype(np.int64)
        head_pos_2d = np.array([(head_bb_int[ 2] + head_bb_int[ 0])/2,(head_bb_int[ 3] + head_bb_int[ 1])/2],dtype='float32')


        body_bb_int = self.body_bb[idx].copy()
        body_bb_int[ [0, 2]] *= img_.shape[2]
        body_bb_int[ [1, 3]] *= img_.shape[1]
        body_bb_int[ 2] += body_bb_int[ 0]
        body_bb_int[ 3] += body_bb_int[ 1]
        body_bb_int = body_bb_int.astype(np.int64)
        body_pos_2d = np.array([(body_bb_int[ 2] + body_bb_int[ 0])/2,(body_bb_int[ 3] + body_bb_int[ 1])/2],dtype='float32')

        #print(self.head_bb[idx])
        #print(self.body_bb[idx])

        #print(head_pos_2d)
        #print(body_pos_2d)
        #print(img_path)



        if self.flip == True:
            gaze = np.array([self.gazes[idx][0]*-1,self.gazes[idx][1]-1,self.gazes[idx][2]], dtype='float32')
            head = np.array([self.heads[idx][0]*-1,self.heads[idx][1]-1,self.heads[idx][2]], dtype='float32')
            body = np.array([self.bodys[idx][0]*-1,self.bodys[idx][1]-1,self.bodys[idx][2]], dtype='float32')

            head_pos_2d = np.array([img_.shape[1] - (head_bb_int[ 2]+head_bb_int[ 0])/2,(head_bb_int[ 3] + head_bb_int[ 1])/2],dtype='float32')

            item = {
                "image":img_,
                "img_path": img_path,
                "head_dir": head,
                "body_dir": body,
                "gaze_dir": gaze,
                "head_pos": self.head_pos[idx],
                "head_pos_2d":head_pos_2d,
                "body_pos_2d":body_pos_2d,
                "head_bb" : head_bb_int,
                "keypoints": np.stack(self.keypoints[idx]).copy()
            }

        else:

            item = {
                "image":img_,
                "img_path": img_path,
                "head_dir": self.heads[idx],
                "body_dir": self.bodys[idx],
                "gaze_dir": self.gazes[idx],
                "head_pos": self.head_pos[idx],
                "head_pos_2d":head_pos_2d,
                "body_pos_2d":body_pos_2d,
                "head_bb" : head_bb_int,
                "keypoints": np.stack(self.keypoints[idx]).copy()
            }
        return item

def create_gafa_dataset(exp_names, root_dir='./data/preprocessed', test=False, augumented=False):
    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]

    dset_list = []
    for ed in exp_dirs:
        cameras = sorted(os.listdir(ed))
        for cm in cameras:
            if not os.path.exists(os.path.join(ed, cm, 'annotation.pickle')):
                continue

            dset = GazeSeqDataset(os.path.join(ed, cm),human=test, flip=False)

            if len(dset) == 0:
                continue
            dset_list.append(dset)

            if augumented:
                ddset = GazeSeqDataset(os.path.join(ed, cm), flip=True)
                if len(ddset) == 0:
                    continue
                dset_list.append(ddset)


    print("in create_gafa_dataset")
    #print(min(len(d) for d  in dset_list))
    res = ConcatDataset(dset_list)
    return res