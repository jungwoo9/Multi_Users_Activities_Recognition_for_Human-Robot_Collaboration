import numpy as np
from pathlib import Path
import pickle

from extract_data_from_db import extract_data_from_db
from normalize_skeleton import normalize_process

import argparse

def extract_single_skeleton_from_directory(path_directory):
    """
    Extracts skeleton data from a directory containing databases.
    """
    skeleton_dict = {}
    for d in sorted(path_directory.iterdir()):
        if d.is_dir():
            for db in d.iterdir():
                if db.is_file():
                    # print(db)
                    _, _, _, skeleton = extract_data_from_db(db)

                    # skip if skeleton is empty
                    if skeleton == []:
                        continue
                    
                    skeleton_dict[db.parts[-2]] = skeleton

                    # skeleton = np.array(skeleton)
                    # print(skeleton.shape)
        
    return skeleton_dict

def extract_single_skeleton_all_joints_from_directory(path_directory):
    """
    Extracts skeleton data with all joints from a directory containing databases.
    """
    skeleton_dict = {}
    for d in sorted(path_directory.iterdir()):
        if d.is_dir():
            for db in d.iterdir():
                if db.is_file():
                    # print(db)
                    _, _, _, skeleton = extract_data_from_db(db, num_joint=32)

                    # skip if skeleton is empty
                    if skeleton == []:
                        continue
                    
                    skeleton_dict[db.parts[-2]] = skeleton

                    # skeleton = np.array(skeleton)
                    # print(skeleton.shape)
        
    return skeleton_dict

def save_single_skeleton_dict(skeleton_dict, path_to_save):
    """
    Saves a dictionary containing skeleton data to a pickle file.
    """
    with open(path_to_save, 'wb') as f:
        pickle.dump(skeleton_dict, f)

def load_single_skeleton_dict(path_to_load):
    """
    Loads a dictionary containing skeleton data from a pickle file.
    """
    with open(path_to_load, 'rb') as f:
        skeleton_dict = pickle.load(f)

    return skeleton_dict

def generate_window(skeleton_dict):
    """
    Generates windows of skeleton data.
    """
    total_samples_per_window = args.window_size
    new_activity_samples_length = args.new_activity_length

    window_skeleton_dict = {}
    for k, v in skeleton_dict.items():
        window = []
        idx = 0
        while idx+total_samples_per_window <= len(v):
            window.append(v[idx: idx+total_samples_per_window])
            idx += new_activity_samples_length
        window_skeleton_dict[k] = window
    
    return window_skeleton_dict

def concatenate_two_skeleton(skeleton_dict):
    """
    Concatenates two skeletons from different participants.
    """
    grouped_skeleton = {}

    ptcp = None

    for k in skeleton_dict.keys():
        print(k)
        ptcp_1 = k.split("_")[0] # participant
        act_1 = k.split("_")[1] # activity
        t_1 = k.split("_")[2] # ignore
        r_1 = k.split("_")[3] # repetition

        if int(t_1[1:]) >= 2:
            continue

        if ptcp is None:
            ptcp = ptcp_1
        
        if int(ptcp_1[1:]) > 4:
            continue

        elif ptcp != ptcp_1:
            path_to_save = f'./data/skeleton/final/grouped/grouped_single_skeleton_dict_{ptcp}.pkl'
            save_single_skeleton_dict(grouped_skeleton, path_to_save)
            ptcp = ptcp_1
            grouped_skeleton = {}

        for k_ in skeleton_dict.keys():
            if ptcp_1 not in k_:
                ptcp_2 = k_.split("_")[0]
                act_2 = k_.split("_")[1]
                t_2 = k_.split("_")[2]
                r_2 = k_.split("_")[3]

                if int(t_2[1:]) >= 2:
                    continue

                new_name = "_".join([ptcp_1, ptcp_2, act_1, act_2, t_1, t_2, r_1, r_2, "skeleton"])
                
                new_skeleton = []
                for sk1 in skeleton_dict[k]:
                    for sk2 in skeleton_dict[k_]:
                        new_skeleton.append(np.concatenate((np.array(sk1),np.array(sk2)), axis=2))
                
                grouped_skeleton[new_name] = new_skeleton

    path_to_save = f'./data/skeleton/final/grouped/grouped_single_skeleton_dict_{ptcp_1}.pkl'
    save_single_skeleton_dict(grouped_skeleton, path_to_save)
        
    return grouped_skeleton

def main():
    if args.step == 'all':
        # extract skeleton from database
        skeleton_dict = extract_single_skeleton_from_directory(Path("./data/decompressed/single_users_decompressed"))

        # save skeleton dictionary file as pickle
        path_to_save = './data/skeleton/raw/single_skeleton_dict.pkl'
        save_single_skeleton_dict(skeleton_dict, path_to_save)

        # normalize skeleton
        norm_skeleton_dict = normalize_process(skeleton_dict)

        # save skeleton dictionary file as pickle
        path_to_save = './data/skeleton/normalized/normalized_single_skeleton_dict.pkl'
        save_single_skeleton_dict(norm_skeleton_dict, path_to_save)

    elif args.step == 'normalise':
        # load skeleton dictionary file from pickle
        path_to_load = './data/skeleton/raw/single_skeleton_dict.pkl'
        skeleton_dict = load_single_skeleton_dict(path_to_load)

        # normalize skeleton
        norm_skeleton_dict = normalize_process(skeleton_dict)

        # save skeleton dictionary file as pickle
        path_to_save = './data/skeleton/normalized/normalized_single_skeleton_dict.pkl'
        save_single_skeleton_dict(norm_skeleton_dict, path_to_save)

    elif args.step == 'generate_window':
        # load skeleton dictionary file from pickle
        path_to_load = './data/skeleton/normalized/normalized_single_skeleton_dict.pkl'
        norm_skeleton_dict = load_single_skeleton_dict(path_to_load)

    elif args.step == "save_raw":
        # skeleton_dict = extract_single_skeleton_all_joints_from_directory(Path("./data/decompressed/single_users_decompressed"))

        # save skeleton dictionary file as pickle
        # path_to_save = './data/skeleton/raw/single_skeleton_all_joints_dict.pkl'
        # save_single_skeleton_dict(skeleton_dict, path_to_save)

        # load skeleton dictionary file from pickle
        path_to_load = './data/skeleton/raw/single_skeleton_all_joints_dict.pkl'
        skeleton_dict = load_single_skeleton_dict(path_to_load)

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
        path_to_save = './data/skeleton/all_joints/single_skeleton_all_joints_dict.pkl'
        save_single_skeleton_dict(new_skeleton_dict, path_to_save)

        return 0
    
    # generate window in each case
    window_skeleton_dict = generate_window(norm_skeleton_dict)

    # concatenate data
    grouped_window_skeleton_dict = concatenate_two_skeleton(window_skeleton_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("step", default='generate_window', choices=['all', 'normalise', 'generate_window', 'save_raw'], help="choose what step to start with | step is extract - normalise - generate_window(including concatenation) | save_raw for visualization preparation")
    parser.add_argument("window_size", default=130, type=int, help="decide window size")
    parser.add_argument("new_activity_length", default=26, type=int, help="decide new activity length to delete those lengh time")

    args = parser.parse_args()

    main()