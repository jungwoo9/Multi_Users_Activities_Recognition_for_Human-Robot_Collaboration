import numpy as np
from pathlib import Path
import sys

def normalize_skeleton(skeleton):
    """
    Normalize a skeleton by removing a specific point (naval) and scaling based on the distance
    between two specified points (naval and neck).
    """
    skeleton_naval = skeleton[1]
    skeleton_neck = skeleton[3]
    
    J0 = np.array([skeleton_naval.x, skeleton_naval.y, skeleton_naval.z])
    J1 = np.array([skeleton_neck.x, skeleton_neck.y, skeleton_neck.z])
    
    normalization = np.linalg.norm(J1-J0)

    skeleton_norm = [[], [], []]
    for i, point in enumerate(skeleton):
        x = point.x
        y = point.y
        z = point.z

        new_x = (x - skeleton_naval.x) / normalization
        new_y = (y - skeleton_naval.y) / normalization
        new_z = (z - skeleton_naval.z) / normalization

        # remove naval point
        if i != 1:
            skeleton_norm[0].append(new_x)
            skeleton_norm[1].append(new_y)
            skeleton_norm[2].append(new_z)

    return skeleton_norm

def calculate_min_max(skeleton_dict):
    """
    Calculate the minimum and maximum values across all skeletons in a dictionary.
    """
    min_ = sys.maxsize
    max_ = -sys.maxsize - 1

    for v in skeleton_dict.values():
        if np.array(v).min() < min_:
            min_ = np.array(v).min()
        if np.array(v).max() > max_:
            max_ = np.array(v).max()

    return min_, max_

def min_max_normalization(skeleton, min_, max_):
    """
    Perform min-max normalization on a skeleton.
    """
    new_max = 1
    new_min = 0
    
    skeleton = np.array(skeleton)
    skeleton = ((skeleton - min_) / (max_ - min_)) * (new_max - new_min) +  new_min
    skeleton = skeleton.tolist()
    
    return skeleton

def normalize_process(skeleton_dict):
    """
    Normalize skeleton by scaling the distance between naval and neck,
    and performing min-max normalization.
    """
    # normalize each skeleton
    norm_skeleton_dict = {}
    for k, v in skeleton_dict.items():
        norm_skeleton_dict[k] = [normalize_skeleton(skeleton) for skeleton in v]
    
    # calculate minimum and maximum value
    min_, max_ = calculate_min_max(norm_skeleton_dict)
    
    # min-max normalize
    for k, v in norm_skeleton_dict.items():
        norm_skeleton_dict[k] = min_max_normalization(v, min_, max_)
    
    return norm_skeleton_dict