import numpy as np
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

import random

def load_dict(path_to_load):
    with open(path_to_load, 'rb') as f:
       dict_ = pickle.load(f)

    return dict_

## augmentation ##
# code referenced by github repository https://github.com/uchidalab/time_series_augmentation/tree/master

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def shear_skeleton(skeleton_data):
    sheared_skeleton = skeleton_data.copy()

    shear_factor = np.random.randint(1, 6) / 10

    H = np.array([[1, shear_factor, 0],
                  [shear_factor, 1, 0],
                  [0, 0, 1]])

    # Apply the shearing transformation
    for i in range(len(skeleton_data)):
        sheared_skeleton[i] = np.dot(H, skeleton_data[i])

    return sheared_skeleton
    
def jitter(x, sigma=0.005):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


## grouped ##

def map_label(label_1, label_2):
    """
    1 : Working
    2 : Requesting
    3 : Preparing
    """
    if label_1==1 and label_2==1:
        return 1 # Working - Working
    elif label_1==1 and label_2==2:
        return 2 # Working - Requesting
    elif label_1==1 and label_2==3:
        return 3 # Working - Preparing
    elif label_1==2 and label_2==1:
        return 4 # Requesting - Working
    elif label_1==2 and label_2==2:
        return 5 # Requesting - Requesting
    elif label_1==2 and label_2==3:
        return 6 # Requesting - Preparing
    elif label_1==3 and label_2==1:
        return 7 # Preparing - Working
    elif label_1==3 and label_2==2:
        return 8 # Preparing - Requesting
    elif label_1==3 and label_2==3:
        return 9 # Preparing - Preparing
    
def split_train_test(skeleton_dict, ptcp_id='s01'):
    train_ = []
    test_ = []

    for k, v in skeleton_dict.items():
        # test
        if ptcp_id in k:
            test_.append(k)
        else:
            train_.append(k)

    return train_, test_

def generate_grouped_data(skeleton_dict, data_):
    data = []
    labels = []
    for k in data_:
        label = map_label(int(k.split("_")[2][1:]), int(k.split("_")[3][1:]))

        sk = np.array(skeleton_dict[k])
        for _ in range(sk.shape[0]):
            labels.append(label)
        data.append(skeleton_dict[k])

    return np.concatenate(data), labels

def update_all_dict(csf3):
    base_dict = {}

    for i in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
        # in case of csf3
        if csf3:
            tmp_dict = load_dict(f"../project/data/grouped/grouped_single_skeleton_dict_{i}.pkl")
            base_dict.update(tmp_dict)
        
        # in case of colab
        else:
            tmp_dict = load_dict(f"/content/drive/MyDrive/3rd_year_project/data/grouped_single_skeleton_dict_{i}.pkl")
            
            # need to use partial data due to the limitation of memory
            sample_dict = {k:tmp_dict[k] for k in random.sample(list(tmp_dict), 350)}
            base_dict.update(sample_dict)
    
    return base_dict
  
def get_grouped_dataloader(ptcp_id='s01', train=True, batch_size=256, csf3=True):
    grouped_window_skeleton_dict = update_all_dict(csf3)

    if ptcp_id == "all":
        data, labels = generate_grouped_data(grouped_window_skeleton_dict, grouped_window_skeleton_dict.keys())
    else:
        train_, test_ = split_train_test(grouped_window_skeleton_dict, ptcp_id)
        
        data, labels = generate_grouped_data(grouped_window_skeleton_dict, train_ if train else test_)
    
    new_data = []
    for d, l in zip(data, labels):
        new_data.append([d, l])
    
    dataloader = torch.utils.data.DataLoader(new_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return dataloader

def get_grouped_by_ptcp_dataloader(ptcp_id='s01', train=True, batch_size=256, csf3=True):
    if csf3:
        grouped_window_skeleton_dict = load_dict(f"../project/data/grouped_by_ptcp/grouped_single_skeleton_dict_{ptcp_id}_by_ptcp.pkl")
    else:
        grouped_window_skeleton_dict = load_dict(f"/content/drive/MyDrive/3rd_year_project/data/grouped_single_skeleton_dict_{ptcp_id}_by_ptcp.pkl")

    data, labels = generate_grouped_data(grouped_window_skeleton_dict, grouped_window_skeleton_dict.keys())
    
    new_data = []

    # uncomment to add data augmentation
    # for d, l in zip(data, labels):
        # d = window_warp(d.reshape(1, 130, -1)).reshape(130, 3, 20)
        # d = shear_skeleton(d)
        # d = jitter(d.reshape(1, 130, -1)).reshape(130, 3, 20)
        # d = window_warp(jitter(d.reshape(1, 130, -1))).reshape(130, 3, 20)
        # new_data.append([d, l])
    
    # for d, l in zip(data, labels):
    #     d = jitter(d.reshape(1, 130, -1)).reshape(130, 3, 20)
    #     new_data.append([d, l])

    for d, l in zip(data, labels):
        new_data.append([d, l])
    
    dataloader = torch.utils.data.DataLoader(new_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return dataloader

## paired ##
    
class PairedSkeletonDataset(Dataset):
    def __init__(self, skeletons, labels, ptcp_ids, ptcp_id='s1', train=True):
        if ptcp_id=="all":
            self.skeletons = skeletons
            self.labels = labels
        else:
            if train:
                idx = np.where([ptcp_id not in x for x in ptcp_ids])
            else:
                idx = np.where([ptcp_id in x for x in ptcp_ids])
            
            self.skeletons = skeletons[idx]
            self.labels = labels[idx]

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        return torch.from_numpy(self.skeletons[idx]), torch.tensor(self.labels[idx])
        
def generate_paired_data(path_skeleton, path_label):
    skeleton_dict = load_dict(path_skeleton)
    label_dict = load_dict(path_label)

    train_skeleton = []
    train_label = []
    train_ptcp_ids = []
    for k_sk, k_lb in zip(skeleton_dict.keys(), label_dict.keys()):
        train_skeleton.extend(skeleton_dict[k_sk])
        train_label.extend(label_dict[k_lb])
        ptcp_id = "_".join(k_sk.split("_")[2:4])
        for _ in range(len(skeleton_dict[k_sk])):
            train_ptcp_ids.append(ptcp_id)

    return np.array(train_skeleton), np.array(train_label), np.array(train_ptcp_ids)
    
def get_paired_dataloader(ptcp_id='s01', train=True, batch_size=256, csf3=True):
    if csf3:
        path_paired_skeleton = "../project/data/paired/paired_window_skeleton_dict.pkl"
        path_paired_label = "../project/data/paired/paired_window_label_dict.pkl"

    else:
        path_paired_skeleton = "/content/drive/MyDrive/3rd_year_project/data/paired_window_skeleton_dict.pkl"
        path_paired_label = "/content/drive/MyDrive/3rd_year_project/data/paired_window_label_dict.pkl"

    skeleton, label, ptcp_ids = generate_paired_data(path_paired_skeleton, path_paired_label)

    dataset = PairedSkeletonDataset(skeleton, label, ptcp_ids, ptcp_id=ptcp_id, train=train)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader
