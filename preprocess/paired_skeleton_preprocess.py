import numpy as np
from pathlib import Path
import pickle

from extract_data_from_db import extract_data_from_db
from normalize_skeleton import normalize_process

from collections import Counter

import argparse

def extract_paired_skeleton_label(path_directory):
    skeleton_dict = {}
    label_dict = {}

    for d in sorted(path_directory.iterdir()):
        if d.is_dir():
            for db in d.iterdir():
                if db.is_file():
                    if 'label' in str(db):
                        _, _, _, label = extract_data_from_db(db)
                        
                        label_dict["_".join(db.parts[-2].split("_")[:-2]+[db.parts[-2].split("_")[-2][-1], db.parts[-2].split("_")[-1]])] = label
                        print(db, len(label))
                    
                    elif 'skeleton' in str(db):
                        _, _, _, skeleton = extract_data_from_db(db)

                        # skip if skeleton is empty
                        if skeleton == []:
                            continue
                        
                        skeleton_dict["_".join(db.parts[-2].split("_")[:-2]+[db.parts[-2].split("_")[-2][-1], db.parts[-2].split("_")[-1]])] = skeleton

                        skeleton = np.array(skeleton)
                        print(db, skeleton.shape)

    return skeleton_dict, label_dict

def extract_paired_skeleton_all_joints_from_directory(path_directory):
    skeleton_dict = {}
    label_dict = {}
    for d in sorted(path_directory.iterdir()):
        if d.is_dir():
            for db in d.iterdir():
                if db.is_file():
                    if 'label' in str(db):
                        _, _, _, label = extract_data_from_db(db)
                        
                        label_dict["_".join(db.parts[-2].split("_")[:-2]+[db.parts[-2].split("_")[-2][-1], db.parts[-2].split("_")[-1]])] = label
                        print(db, len(label))
                    
                    elif 'skeleton' in str(db):
                        _, _, _, skeleton = extract_data_from_db(db, num_joint=32)

                        # skip if skeleton is empty
                        if skeleton == []:
                            continue
                        
                        skeleton_dict["_".join(db.parts[-2].split("_")[:-2]+[db.parts[-2].split("_")[-2][-1], db.parts[-2].split("_")[-1]])] = skeleton

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

def identify_ptcp(skeleton_dict):
    ptcp_dict = {}
    for k in skeleton_dict.keys():
        new_ptcps = []
        for sk in skeleton_dict[k]:
            if sk[0].x > 0:
                new_ptcps.append(200)
            elif sk[0].x < 0:
                new_ptcps.append(100)
        ptcp_dict[k] = new_ptcps

    return ptcp_dict

def concatenate_two_skeleton(skeleton_dict, label_dict, ptcp_dict):
    new_skeleton_dict = {}
    new_label_dict = {}

    for k_sk, k_lb in zip(skeleton_dict.keys(), label_dict.keys()):
        skeleton, label, ptcp = skeleton_dict[k_sk], label_dict[k_lb], ptcp_dict[k_sk]

        first_ptcp = ptcp[0]
        tmp_ptcp = ptcp[0]

        buf_100_skeleton = []
        buf_100_label = []
        buf_200_skeleton = []
        buf_200_label = []

        length = min(len(skeleton), len(label))

        new_skeleton = []
        new_label = []

        for i in range(length):
            if ptcp[i] == first_ptcp and i != 0 and tmp_ptcp != ptcp[i]:
                for sk1 in range(len(buf_100_skeleton)):
                    for sk2 in range(len(buf_200_skeleton)):
                        if buf_100_label[sk1] == buf_200_label[sk2]:
                            new_skeleton.append(np.concatenate((np.array(buf_100_skeleton[sk1]), np.array(buf_200_skeleton[sk2])), axis=1))
                            new_label.append(buf_100_label[sk1])

                buf_100_skeleton = []
                buf_100_label = []
                buf_200_skeleton = []
                buf_200_label = []

            if ptcp[i] == 100:
                buf_100_skeleton.append(skeleton[i])
                buf_100_label.append(label[i])

            elif ptcp[i] == 200:
                buf_200_skeleton.append(skeleton[i])
                buf_200_label.append(label[i])

            tmp_ptcp = ptcp[i]

        new_skeleton_dict[k_sk] = new_skeleton
        new_label_dict[k_lb] = new_label

    return new_skeleton_dict, new_label_dict

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
                
                new_skeleton.extend(skeleton[start_cut:i-args.new_activity_length])
                new_label.extend(label[start_cut:i-args.new_activity_length])
                start_cut = i + args.new_activity_length
        
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
        for i in range(len(skeleton)-args.window_size+1):
            if Counter(label[i:i+args.window_size])[label[i+args.window_size-1]] < 70:
                continue

            skeleton_window.append(skeleton[i:i+args.window_size])
            label_window.append(label[i+args.window_size-1])
        
        skeleton_window_dict[k_sk] = skeleton_window
        label_window_dict[k_lb] = label_window
    
    return skeleton_window_dict, label_window_dict

def main():
    if args.step == 'all':
        # check participants id correctness
        path_directory = Path("./data/decompressed/paired_users_decompressed")

        # get paired skeleton
        skeleton_dict, label_dict = extract_paired_skeleton_label(path_directory)

        # save skeleton and label dictionary
        path_to_save_skeleton = './data/skeleton/raw/paired_skeleton_dict.pkl'
        path_to_save_label = './data/skeleton/raw/paired_label_dict.pkl'
        save_dict(skeleton_dict, path_to_save_skeleton)
        save_dict(label_dict, path_to_save_label)

        # identify participants as 100 and 200
        ptcp_dict = identify_ptcp(skeleton_dict)

        # normalize skeleton
        norm_skeleton_dict = normalize_process(skeleton_dict)

        # save normalized skeleton dictionary and label dictionary
        path_to_save_skeleton = './data/skeleton/normalized/normalized_paired_skeleton_dict.pkl'
        path_to_save_label = './data/skeleton/normalized/paired_label_dict.pkl'
        path_to_save_ptcp = './data/skeleton/normalized/participant_ids.pkl'
        save_dict(norm_skeleton_dict, path_to_save_skeleton)
        save_dict(label_dict, path_to_save_label)
        save_dict(ptcp_dict, path_to_save_ptcp)

    elif args.step == 'normalise':
        # load raw skeleton and label dictionary
        path_to_load_skeleton = './data/skeleton/raw/paired_skeleton_dict.pkl'
        path_to_load_label = './data/skeleton/raw/paired_label_dict.pkl'
        skeleton_dict = load_dict(path_to_load_skeleton)
        label_dict = load_dict(path_to_load_label)

        # identify participants as 100 and 200
        ptcp_dict = identify_ptcp(skeleton_dict)

        # normalize skeleton
        norm_skeleton_dict = normalize_process(skeleton_dict)

        # save normalized skeleton dictionary and label dictionary
        path_to_save_skeleton = './data/skeleton/normalized/normalized_paired_skeleton_dict.pkl'
        path_to_save_label = './data/skeleton/normalized/paired_label_dict.pkl'
        path_to_save_ptcp = './data/skeleton/normalized/participant_ids.pkl'
        save_dict(norm_skeleton_dict, path_to_save_skeleton)
        save_dict(label_dict, path_to_save_label)
        save_dict(ptcp_dict, path_to_save_ptcp)

    elif args.step == 'generate_window':
        # load normalised skeleton, label, and ptcp id dictionary
        path_to_load_skeleton = './data/skeleton/normalized/normalized_paired_skeleton_dict.pkl'
        path_to_load_label = './data/skeleton/normalized/paired_label_dict.pkl'
        path_to_load_ptcp = './data/skeleton/normalized/participant_ids.pkl'
        norm_skeleton_dict = load_dict(path_to_load_skeleton)
        label_dict = load_dict(path_to_load_label)
        ptcp_dict = load_dict(path_to_load_ptcp)

    elif args.step == "save_raw":
        skeleton_dict, _ = extract_paired_skeleton_all_joints_from_directory(Path("./data/decompressed/paired_users_decompressed"))

        # save skeleton dictionary file as pickle
        path_to_save = './data/skeleton/raw/paired_skeleton_all_joints_dict.pkl'
        save_dict(skeleton_dict, path_to_save)

        # load skeleton dictionary file from pickle
        path_to_load = './data/skeleton/raw/paired_skeleton_all_joints_dict.pkl'
        skeleton_dict = load_dict(path_to_load)

        new_skeleton_dict = {}
        for k in skeleton_dict.keys():
            new_skeletons = []
            for sk in skeleton_dict[k]:
                x = []
                y = []
                z = []
                for node in sk:
                    x.append(node.x)
                    y.append(node.y)
                    z.append(node.z)
                new_skeletons.append([x, y, z])
            new_skeleton_dict[k] = new_skeletons

        # save skeleton dictionary file as pickle
        path_to_save = './data/skeleton/all_joints/paired_skeleton_all_joints_dict.pkl'
        save_dict(new_skeleton_dict, path_to_save)

        return 0
    
    # concatenate two participants and then remove near new activity
    concat_skeleton_dict, concat_label_dict = remove_near_new_activity(*concatenate_two_skeleton(norm_skeleton_dict, label_dict, ptcp_dict))
    
    # generate window
    skeleton_window_dict, label_window_dict = generate_window(concat_skeleton_dict, concat_label_dict)

    # save paired skeleton and label dictionary
    path_to_save_skeleton = './data/skeleton/final/paired/paired_window_skeleton_dict.pkl'
    path_to_save_label = './data/skeleton/final/paired/paired_window_label_dict.pkl'
    save_dict(skeleton_window_dict, path_to_save_skeleton)
    save_dict(label_window_dict, path_to_save_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("step", default='generate_window', choices=['all', 'normalise', 'generate_window', 'save_raw'], help="choose what step to start with | step is extract - normalise - generate_window(including concatenation) | save_raw for visualization preparation")
    parser.add_argument("window_size", default=130, type=int, help="decide window size")
    parser.add_argument("new_activity_length", default=52, type=int, help="decide new activity length to delete those lengh time")

    args = parser.parse_args()

    main()