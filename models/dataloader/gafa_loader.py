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
        self.bodys = anno_data["bodys"]
        self.heads = anno_data["heads"]
        self.gazes = anno_data["gazes"]
        self.R_cam = anno_data["R_cam"]
        self.t_cam = anno_data["t_cam"]
        self.body_pos = anno_data["body_pos"]
        self.head_pos = anno_data["head_pos"]
        self.img_index = anno_data['index']
        self.keypoints = anno_data['keypoins']
        #self.head_bb = np.vstack(anno_data['head_bb']).astype(np.float32)
        #self.body_bb = np.vstack(anno_data['body_bb']).astype(np.float32)
                

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
        img_ = transform(img)


        item = {
            "image":img_,
            "img_path": img_path,
            "head_dir": self.heads[idx],
            "body_dir": self.bodys[idx],
            "gaze_dir": self.gazes[idx],
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