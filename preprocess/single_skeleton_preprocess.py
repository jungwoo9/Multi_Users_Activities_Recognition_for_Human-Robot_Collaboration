import numpy as np
from pathlib import Path
import pickle

from extract_data_from_db import extract_data_from_db
from normalize_skeleton import normalize_process

def extract_single_skeleton_from_directory(path_directory):
    
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

def save_single_skeleton_dict(skeleton_dict, path_to_save):
    with open(path_to_save, 'wb') as f:
        pickle.dump(skeleton_dict, f)

def load_single_skeleton_dict(path_to_load):
    with open(path_to_load, 'rb') as f:
        skeleton_dict = pickle.load(f)

    return skeleton_dict

def generate_window(skeleton_dict):
    total_samples_per_window = 130
    new_activity_samples_length = 26

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

    paired_skeleton = {}
    for k in skeleton_dict.keys():
        print(k)
        ptcp_1 = k.split("_")[0] # participant
        act_1 = k.split("_")[1] # activity
        r_1 = int(k.split("_")[3][1:]) # repetition

        for k_ in skeleton_dict.keys():
            if ptcp_1 not in k_:
                ptcp_2 = k_.split("_")[0]
                act_2 = k_.split("_")[1]
                r_2 = int(k_.split("_")[3][1:])
                new_name = "_".join([ptcp_1, ptcp_2, act_1, act_2, str(r_1 + r_2), "skeleton"])

                # use minimum window size
                window_size = min(np.array(skeleton_dict[k]).shape[0], np.array(skeleton_dict[k_]).shape[0])
                new_skeleton = np.concatenate((np.array(skeleton_dict[k])[:window_size], np.array(skeleton_dict[k_])[:window_size]), axis=3)

                paired_skeleton[new_name] = new_skeleton

    return paired_skeleton

if __name__ == "__main__":
    # extract skeleton from database
    # skeleton_dict = extract_single_skeleton_from_directory(Path("./data/decompressed/single_users_decompressed"))

    # save skeleton dictionary file as pickle
    # path_to_save = './data/skeleton/raw/single_skeleton_dict.pkl'
    # save_single_skeleton_dict(skeleton_dict, path_to_save)

    # load skeleton dictionary file from pickle
    # path_to_load = './data/skeleton/raw/single_skeleton_dict.pkl'
    # skeleton_dict = load_single_skeleton_dict(path_to_load)

    # normalize skeleton
    # norm_skeleton_dict = normalize_process(skeleton_dict)

    # save skeleton dictionary file as pickle
    # path_to_save = './data/skeleton/normalized/normalized_single_skeleton_dict.pkl'
    # save_single_skeleton_dict(norm_skeleton_dict, path_to_save)

    # load skeleton dictionary file from pickle
    path_to_load = './data/skeleton/normalized/normalized_single_skeleton_dict.pkl'
    norm_skeleton_dict = load_single_skeleton_dict(path_to_load)
    
    # generate window in each case
    window_skeleton_dict = generate_window(norm_skeleton_dict)

    # concatenate data
    paired_window_skeleton_dict = concatenate_two_skeleton(window_skeleton_dict)

    # save paired skeleton
    path_to_save = './data/skeleton/final/paired_single_skeleton_dict.pkl'
    save_single_skeleton_dict(paired_window_skeleton_dict, path_to_save)
