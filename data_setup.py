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
