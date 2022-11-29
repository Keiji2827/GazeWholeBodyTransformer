import os
import pickle
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

class GazeSeqDataset(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path

        # load annotation
        with open(os.path.join(video_path, 'annotation.pickle'), "rb") as f:
            anno_data = pickle.load(f)

        self.bodys = anno_data["bodys"]
        self.heads = anno_data["heads"]
        self.gazes = anno_data["gazes"]
        self.R_cam = anno_data["R_cam"]
        self.t_cam = anno_data["t_cam"]
        self.body_pos = anno_data["body_pos"]
        self.head_pos = anno_data["head_pos"]
        self.img_index = anno_data['index']
        self.keypoints = anno_data['keypoins']

    def __len__(self):
        return len(self.gazes)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} >= len {len(self)}")

        idx = self.img_index[idx]

        img = cv2.imread(os.path.join(self.video_path, f"{self.img_index[idx]:06}.jpg"))[:,:,::-1]

        item = {
            "image":img,
            "head_dir": self.heads[idx],
            "body_dir": self.bodys[idx],
            "gaze_dir": self.gazes[idx],
            "keypoints": self.keypoints[idx]
        }

        return item

def create_gafa_dataset(exp_names, root_dir='./data/preprocessed'):
    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]

    dset_list = []
    for ed in exp_dirs:
        cameras = sorted(os.listdir(ed))
        for cm in cameras:
            if not os.path.exists(os.path.join(ed, cm, 'annotation.pickle')):
                continue

            dset = GazeSeqDataset(os.path.join(ed, cm))

            if len(dset) == 0:
                continue

            dset_list.append(dset)

    return ConcatDataset(dset_list)