from torch.utils.data import Dataset

import cv2
import os
import numpy as np
from functools import cmp_to_key

def model_name_compare(x, y):
    x = deal_name(x)
    y = deal_name(y)
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0

def deal_name(s):
    s = s.split('.')[0]
    # names = s.split('_')
    # print(names)
    # return int(names[-2])*100 + int(names[-1])
    return int(s)
        

def make_dataset(dataset_dir):
    frame_path = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(sorted(os.listdir(dataset_dir), key = cmp_to_key(model_name_compare))):
        # clipsFolderPath = os.path.join(dataset_dir, folder, 'ground_truth')
        clipsFolderPath = os.path.join(dataset_dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        frame_path.append([])
        # Find and loop over all the frames inside the clip.
        
        for image in sorted(os.listdir(clipsFolderPath), key = cmp_to_key(model_name_compare)): #这里不排序的话，序列就被打断了
        # for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            
            # if int(image.split('.')[0]) <41:
            #     print(image)
            frame_path[index].append(os.path.join(clipsFolderPath, image))
    # print(frame_path)       
    return frame_path

    
def make_dataset_train(dataset_dir):
    frame_path = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(sorted(os.listdir(dataset_dir), key = cmp_to_key(model_name_compare))):
        # print(index)
        if index == 10000:
            break
        # clipsFolderPath = os.path.join(dataset_dir, folder, 'ground_truth')
        clipsFolderPath = os.path.join(dataset_dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        frame_path.append([])
        # Find and loop over all the frames inside the clip.

        for image in sorted(os.listdir(clipsFolderPath), key = cmp_to_key(model_name_compare)): #这里不排序的话，序列就被打断了
        # for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            
            # if int(image.split('.')[0]) <41:
            #     print(image)
            frame_path[index].append(os.path.join(clipsFolderPath, image))
            
    return frame_path

class Radar(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset(dataset_dir)
        # print(len(self.frame_path))
        self.seq_len = seq_len
        self.train = train
        # print(seq_len)
        self.clips = []
        for video_i in range(len(self.frame_path)):
            video_frame_num = len(self.frame_path[video_i])
            # print(video_frame_num)
            # print(video_frame_num - seq_len + 1)
            self.clips += [(video_i, t) for t in range(video_frame_num - seq_len + 1)] if train \
                else [(video_i, t * seq_len) for t in range(video_frame_num // seq_len)]
        # print(len(self.clips))

    def __getitem__(self, idx):
        (video_idx, data_start) = self.clips[idx]
        sample = []
        for frame_range_i in range(data_start, data_start+self.seq_len):
            frame = cv2.imread(self.frame_path[video_idx][frame_range_i], cv2.IMREAD_GRAYSCALE)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32)
            frame = frame/255.0
            sample.append(frame)
        return sample

    def __len__(self):
        return len(self.clips)

class Radar_train(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset_train(dataset_dir)
        self.seq_len = seq_len
        self.train = train
        # print(len(self.frame_path))
        self.clips = []
        for video_i in range(len(self.frame_path)):
            video_frame_num = len(self.frame_path[video_i])
            self.clips += [(video_i, t) for t in range(video_frame_num - seq_len + 1)] if train \
                else [(video_i, t * seq_len) for t in range(video_frame_num // seq_len)]

    def __getitem__(self, idx):
        (video_idx, data_start) = self.clips[idx]
        sample = []
        for frame_range_i in range(data_start, data_start+self.seq_len):
            frame = cv2.imread(self.frame_path[video_idx][frame_range_i], cv2.IMREAD_GRAYSCALE)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32)
            frame = frame/255.0
            sample.append(frame)
        return sample

    def __len__(self):
        return len(self.clips)

class Satellite(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset(dataset_dir)
        self.seq_len = seq_len
        self.train = train
        # print(len(self.frame_path))
        self.clips = []
        for video_i in range(len(self.frame_path)):
            video_frame_num = len(self.frame_path[video_i])
            self.clips += [(video_i, t) for t in range(video_frame_num - seq_len + 1)] if train \
                else [(video_i, t * seq_len) for t in range(video_frame_num // seq_len)]

    def __getitem__(self, idx):
        (video_idx, data_start) = self.clips[idx]
        sample = []
        for frame_range_i in range(data_start, data_start+self.seq_len):
            frame = cv2.imread(self.frame_path[video_idx][frame_range_i], cv2.IMREAD_GRAYSCALE)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32)
            frame = frame/255.0
            sample.append(frame)
        return sample

    def __len__(self):
        return len(self.clips)

