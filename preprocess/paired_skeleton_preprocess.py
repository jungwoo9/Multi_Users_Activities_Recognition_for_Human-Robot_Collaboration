import numpy as np
from pathlib import Path
import pickle

from extract_data_from_db import extract_data_from_db
from normalize_skeleton import normalize_process

from collections import Counter

def validate_participant_ids(path_directory):
    """
    Validates the correctness of research participant IDs (object IDs) in pair data markers (skeleton directories).
    IDs must be either 100 or 200; otherwise, it indicates a reassignment error.
    """
    for directory in sorted(path_directory.iterdir()):
        if directory.is_dir():
            if str(directory).split("_")[-1] == "skeleton":
                for db in directory.iterdir():
                    if db.is_file():
                        _, _, _, data = extract_data_from_db(db)
                        for i in data:
                            if i[-1] not in [100, 200]:
                                print("Include new id(reassigend):", i[-1])

def extract_paired_skeleton_label(path_directory):
    skeleton_dict = {}
    label_dict = {}

    for d in sorted(path_directory.iterdir()):
        if d.is_dir():
            for db in d.iterdir():
                if db.is_file():
                    if 'label' in str(db):
                        _, _, _, label = extract_data_from_db(db)
                        label_dict[db.parts[-2]] = label
                        print(db, len(label))
                    
                    elif 'skeleton' in str(db):
                        _, _, _, skeleton = extract_data_from_db(db)

                        # skip if skeleton is empty
                        if skeleton == []:
                            continue
                        
                        skeleton_dict[db.parts[-2]] = skeleton

                        skeleton = np.array(skeleton)
                        print(db, skeleton.shape)

    return skeleton_dict, label_dict

def save_dict(dict_, path_to_save):
    """
    Saves a dictionary to a pickle file.
    """
    with open(path_to_save, 'wb') as f:
        pickle.dump(dict_, f)

def load_dict(path_to_load):
    """
    Loads a dictionary from a pickle file.
    """
    with open(path_to_load, 'rb') as f:
        dict_ = pickle.load(f)

    return dict_

def remove_consecutive_duplicates(skeleton_dict, label_dict):
    """
    Removes consecutive duplicates.
    For example, if the participants skeleton is like this
        100 200 200 100 100 100 100 200
        then
        100 200 100 200
    """

    new_skeleton_dict = {}
    new_label_dict = {}

    for k_sk, k_lb in zip(skeleton_dict.keys(), label_dict.keys()):
        skeleton = skeleton_dict[k_sk]
        label = label_dict[k_lb]

        temp_id = -100 # store the temporal id
        new_skeleton = []
        new_label = []

        length = len(skeleton) if len(skeleton) <= len(label) else len(label)
        for i in range(length):
            id = skeleton[i][-1] # participant id was saved and appended at the end of list
            s = skeleton[i][:-1] # get only skeleton part without id
            l = label[i]

            # check if the current ID is different from the previous one
            if id != temp_id:
                temp_id = id
                new_skeleton.append(s)
                new_label.append(l)
            
            # update the last element in the list (overwrite if ID is the same)
            new_skeleton[-1] = s
            new_label[-1] = l
        
        new_skeleton_dict[k_sk] = new_skeleton
        new_label_dict[k_lb] = new_label
    
    return new_skeleton_dict, new_label_dict

def concatenate_two_skeleton(skeleton_dict, label_dict):
    concat_skeleton_dict = {}
    concat_label_dict = {}

    for k_sk, k_lb in zip(skeleton_dict.keys(), label_dict.keys()):
        skeleton, label = skeleton_dict[k_sk], label_dict[k_lb]

        # remove the last element
        if len(skeleton) % 2 != 0:
            skeleton = skeleton[:-1]
            label = label[:-1]
        
        # concatenate pair participants
        concat_skeleton_lst = []
        concat_label_lst = []

        i = 0
        for _ in range(int(len(skeleton)/2)):
            # if two pair participants have different label, delete it
            if label[i] != label[i+1]:
                i += 2
                continue
            concat_skeleton = np.concatenate((skeleton[i], skeleton[i+1]), axis=1)
            concat_skeleton_lst.append(concat_skeleton.tolist())
            concat_label_lst.append(label[i])
            i += 2

        concat_skeleton_dict[k_sk] = concat_skeleton_lst
        concat_label_dict[k_lb] = concat_label_lst

    return concat_skeleton_dict, concat_label_dict

def remove_near_new_activity(skeleton_dict, label_dict):

    new_skeleton_dict = {}
    new_label_dict = {}

    for k_sk, k_lb in zip(skeleton_dict.keys(), label_dict.keys()):
        skeleton, label = skeleton_dict[k_sk], label_dict[k_lb]
        
        new_skeleton = []
        new_label = []
        start_cut = 0
        tmp_label = label[0]
        for i in range(len(skeleton)):
            if tmp_label != label[i]:
                tmp_label = label[i]
                
                new_skeleton.extend(skeleton[start_cut:i-6])
                new_label.extend(label[start_cut:i-6])
                start_cut = i + 6
        
        new_skeleton.extend(skeleton[start_cut:])
        new_label.extend(label[start_cut:])

        new_skeleton_dict[k_sk] = new_skeleton
        new_label_dict[k_lb] = new_label
        
    return new_skeleton_dict, new_label_dict


def generate_window(skeleton_dict, label_dict):
    skeleton_window_dict = {}
    label_window_dict = {}
    for k_sk, k_lb in zip(skeleton_dict.keys(), label_dict.keys()):
        skeleton, label = skeleton_dict[k_sk], label_dict[k_lb]
        
        skeleton_window = []
        label_window = []
        for i in range(len(skeleton)-130+1):
            skeleton_window.append(skeleton[i:i+130])
            label_window.append(label[i+129])
        
        skeleton_window_dict[k_sk] = skeleton_window
        label_window_dict[k_lb] = label_window
    
    return skeleton_window_dict, label_window_dict

if __name__ == "__main__":
    # check participants id correctness
    # path_directory = Path("./data/decompressed/paired_users_decompressed")
    # validate_participant_ids(path_directory)

    # get paired skeleton
    # skeleton_dict, label_dict = extract_paired_skeleton_label(path_directory)

    # save skeleton and label dictionary
    # path_to_save_skeleton = './data/skeleton/raw/paired_skeleton_dict.pkl'
    # path_to_save_label = './data/skeleton/raw/paired_label_dict.pkl'
    # save_dict(skeleton_dict, path_to_save_skeleton)
    # save_dict(label_dict, path_to_save_label)

    # load skeleton and label dictionary
    path_to_load_skeleton = './data/skeleton/raw/paired_skeleton_dict.pkl'
    path_to_load_label = './data/skeleton/raw/paired_label_dict.pkl'
    skeleton_dict = load_dict(path_to_load_skeleton)
    label_dict = load_dict(path_to_load_label)

    # remove consecutive duplicates
    skeleton_dict, label_dict = remove_consecutive_duplicates(skeleton_dict, label_dict)

    # normalize skeleton
    norm_skeleton_dict = normalize_process(skeleton_dict)

    # save normalized skeleton dictionary and label dictionary that removed consecutive duplicates
    path_to_save_skeleton = './data/skeleton/normalized/normalized_paired_skeleton_dict.pkl'
    path_to_save_label = './data/skeleton/normalized/removed_duplicates_paired_label_dict.pkl'
    save_dict(norm_skeleton_dict, path_to_save_skeleton)
    save_dict(label_dict, path_to_save_label)

    # concatenate two participants and then remove near new activity
    concat_skeleton_dict, concat_label_dict = remove_near_new_activity(*concatenate_two_skeleton(norm_skeleton_dict, label_dict))
    
    # generate window
    skeleton_window_dict, label_window_dict = generate_window(concat_skeleton_dict, concat_label_dict)

    # save paired skeleton and label dictionary
    path_to_save_skeleton = './data/skeleton/final/paired_window_skeleton_dict.pkl'
    path_to_save_label = './data/skeleton/final/paired_window_label_dict.pkl'
    save_dict(skeleton_window_dict, path_to_save_skeleton)
    save_dict(label_window_dict, path_to_save_label)