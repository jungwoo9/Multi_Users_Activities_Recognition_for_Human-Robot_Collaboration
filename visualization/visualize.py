from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from mpl_toolkits import mplot3d
import pickle

from PIL import Image, ImageTk

def label_to_works(label):
    if label == 1:
        return "Working", "Working"
    elif label == 2:
        return "Working", "Requesting"
    elif label == 3:
        return "Working", "Preparing"
    elif label == 4:
        return "Requesting", "Working"
    elif label == 5:
        return "Requesting", "Requesting"
    elif label == 6:
        return "Requesting", "Preparing"
    elif label == 7:
        return "Preparing", "Working"
    elif label == 8:
        return "Preparing", "Requesting"
    elif label == 9:
        return "Preparing", "Preparing"

def works_to_label(label_1, label_2):
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

def load_skeleton_dict():
    global raw_norm, single_paired, grouped_paired
    
    if raw_norm.get() == "raw":
        if single_paired.get() == "single":
            with open("./data/skeleton/all_joints/single_skeleton_all_joints_dict.pkl", "rb") as f:
                skeleton_dict = pickle.load(f)
        else:
            with open("./data/skeleton/all_joints/paired_skeleton_all_joints_dict.pkl", "rb") as f:
                skeleton_dict = pickle.load(f)
            with open("./data/skeleton/raw/paired_label_dict.pkl", "rb") as f:
                label_dict = pickle.load(f)

            return skeleton_dict, label_dict

    elif raw_norm.get() == "norm":
        if grouped_paired.get() == "grouped":
            # only show participant 1
            with open("./data/skeleton/final/grouped/grouped_single_skeleton_dict_s01.pkl", "rb") as f:
                skeleton_dict = pickle.load(f)
        
        else:
            with open("./data/skeleton/final/paired/paired_window_skeleton_dict.pkl", "rb") as f:
                skeleton_dict = pickle.load(f)
            with open("./data/skeleton/final/paired/paired_window_label_dict.pkl", "rb") as f:
                label_dict = pickle.load(f)
    
            return skeleton_dict, label_dict
        
    else:
        if both_dataset.get() == "grouped":
            with open("./data/skeleton/all_joints/single_skeleton_all_joints_dict.pkl", "rb") as f:
                skeleton_dict_raw = pickle.load(f)

            return skeleton_dict_raw
        else:
            with open("./data/skeleton/all_joints/paired_skeleton_all_joints_dict.pkl", "rb") as f:
                skeleton_dict = pickle.load(f)
            with open("./data/skeleton/raw/paired_label_dict.pkl", "rb") as f:
                label_dict = pickle.load(f)
            
            return skeleton_dict, label_dict
    
    return skeleton_dict, None

def normalize_skeleton(skeleton):
    """
    Normalize a skeleton by removing a specific point (naval) and scaling based on the distance
    between two specified points (naval and neck).
    """
    skeleton_naval = skeleton[1]
    skeleton_neck = skeleton[3]
    
    J0 = np.array([skeleton_naval[0], skeleton_naval[1], skeleton_naval[2]])
    J1 = np.array([skeleton_neck[0], skeleton_neck[1], skeleton_neck[2]])
    
    normalization = np.linalg.norm(J1-J0)

    skeleton_norm = [[], [], []]
    for i, point in enumerate(skeleton):
        x = point[0]
        y = point[1]
        z = point[2]

        new_x = (x - skeleton_naval[0]) / normalization
        new_y = (y - skeleton_naval[1]) / normalization
        new_z = (z - skeleton_naval[2]) / normalization

        # remove naval point
        if i != 1:
            skeleton_norm[0].append(new_x)
            skeleton_norm[1].append(new_y)
            skeleton_norm[2].append(new_z)
    
    return np.array(skeleton_norm)

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

def normalize_skeletons(skeletons):
    idx = [0, 1, 2, 3, 5, 6, 8, 12, 13, 15, 26]
    
    norm_skeletons = []
    min_ = []
    max_ = []
    for i in range(len(skeletons)):
        sk = normalize_skeleton(skeletons[i, :, idx])
        norm_skeletons.append(sk)
        min_.append(sk.min())
        max_.append(sk.max())

    for i in range(len(skeletons)):
        norm_skeletons[i] = min_max_normalization(norm_skeletons[i], min(min_), max(max_))

    return np.array(norm_skeletons)

def get_single_skeleton(skeleton_dict):
    global raw_norm, actions
    if actions.get() == "working":
        a = "a01"
    elif actions.get() == "requesting":
        a = "a02"
    elif actions.get() == "preparing":
        a = "a03"
    elif actions.get() == "all":
        a = "a01"
        a2 = "a02"
        a3 = "a03"
    
    if raw_norm.get() == 'raw' or raw_norm.get() == 'both':
        samples = [k for k in skeleton_dict.keys() if a in k]
        if actions.get() == "all":
            samples2 = [k for k in skeleton_dict.keys() if a2 in k]
            samples3 = [k for k in skeleton_dict.keys() if a3 in k]
    else:
        samples = [k for k in skeleton_dict.keys() if a == k.split("_")[2]]
        if actions.get() == "all":
            samples2 = [k for k in skeleton_dict.keys() if a2 == k.split("_")[2]]
            samples3 = [k for k in skeleton_dict.keys() if a3 == k.split("_")[2]]

    sample = skeleton_dict[samples[np.random.randint(len(samples))]]
    sample = sample[np.random.randint(len(sample))]

    if actions.get() == "all":
        sample2 = skeleton_dict[samples2[np.random.randint(len(samples2))]]
        sample2 = sample2[np.random.randint(len(sample2))]
        sample3 = skeleton_dict[samples3[np.random.randint(len(samples3))]]
        sample3 = sample3[np.random.randint(len(sample3))]

    # choose one random index from time window
    if raw_norm.get() == "norm":
        sample = sample[np.random.randint(len(sample))]
        if actions.get() == "all":
            sample2 = sample2[np.random.randint(len(sample2))]
            sample3 = sample3[np.random.randint(len(sample3))]

    return np.array(sample) if actions.get() != "all" else np.array([sample, sample2, sample3])

def get_paired_skeleton(skeleton_dict, label_dict):
    global raw_norm, labels

    # get random skeleton and label dictionary
    dict_name = list(skeleton_dict.keys())[np.random.randint(len(skeleton_dict.keys()))]
    sample_sk = skeleton_dict[dict_name]
    sample_lb = label_dict["_".join(dict_name.split("_")[:-1]+['label'])]
    min_len = min(len(sample_sk), len(sample_lb))
    
    # get skeleton and label which match to label seletected from user
    sample_sk = np.array(sample_sk)[:min_len]
    sample_lb = np.array(sample_lb)[:min_len]
    sample_sk = sample_sk[np.where(sample_lb == int(labels.get()))[0]]
    sample_lb = sample_lb[np.where(sample_lb == int(labels.get()))[0]]

    if raw_norm.get() == "raw" or raw_norm.get() == "both":
        # divide to two participant based on location
        minus_sample = []
        plus_sample = []
        for i in range(len(sample_lb)):
            if sample_sk[i][0].sum() < 0:
                minus_sample.append(sample_sk[i])
            else:
                plus_sample.append(sample_sk[i])
            
        # get random skeleton
        min_len = min(len(minus_sample), len(plus_sample))
        idx = np.random.randint(min_len)

        return minus_sample[idx], plus_sample[idx]
    
    elif raw_norm.get() == "norm":
        # choose window
        sample_sk = sample_sk[np.random.randint(len(sample_sk))]

        # choose one skeleton in window(among 130)
        sample_sk = sample_sk[np.random.randint(len(sample_sk))]
        return sample_sk, None

    return 0
    
def plot_skeleton(skeleton, label, norm=True):
    global right_frame
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')

    mins, maxs = skeleton.min(axis=1), skeleton.max(axis=1)

    num_sk = 10 if norm else 32
    for i in range(num_sk):
        ax.scatter(skeleton[0][i], skeleton[2][i], skeleton[1][i], c='r')
    
    if norm:    
        connections = [[0,1], [1,2], [1,3], [3,4], [4,5], [1,6], [6,7], [7,8], [2,9]]
    else:
        connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11], [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19,20], [20,21], [0,22], [22,23], [23,24], [24,25], [3,26], [26,27], [26, 28], [26,29], [26,30], [26,31]]

    for i, j in connections:
        ax.plot([skeleton[0][i], skeleton[0][j]], [skeleton[2][i], skeleton[2][j]], [skeleton[1][i], skeleton[1][j]], c='r')
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[2], maxs[2]])
    ax.set_zlim([mins[1], maxs[1]])
    ax.invert_zaxis()
    plt.title(label)
    
    canvas = FigureCanvasTkAgg(fig, 
                               master = right_frame)
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   right_frame) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack()

def plot_both_single_skeleton(sample, norm_sample, label):
    global right_frame

    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    fig.suptitle(label)

    for ax, skeleton, norm in zip([ax1, ax2], [sample, norm_sample], [False, True]):
        mins, maxs = skeleton.min(axis=1), skeleton.max(axis=1)

        num_sk = 10 if norm else 32
        for i in range(num_sk):
            ax.scatter(skeleton[0][i], skeleton[2][i], skeleton[1][i], c='r')
        
        if norm:    
            connections = [[0,1], [1,2], [1,3], [3,4], [4,5], [1,6], [6,7], [7,8], [2,9]]
        else:
            connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11], [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19,20], [20,21], [0,22], [22,23], [23,24], [24,25], [3,26], [26,27], [26, 28], [26,29], [26,30], [26,31]]

        for i, j in connections:
            ax.plot([skeleton[0][i], skeleton[0][j]], [skeleton[2][i], skeleton[2][j]], [skeleton[1][i], skeleton[1][j]], c='r')
            
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[2], maxs[2]])
        ax.set_zlim([mins[1], maxs[1]])
        ax.invert_zaxis()
        ax.set_title("normalized skeleton" if norm else "raw skeleton")
    
    canvas = FigureCanvasTkAgg(fig, 
                               master = right_frame)
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   right_frame) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack()

def plot_all_skeletons(samples, norm=True):
    global right_frame

    fig = plt.figure(figsize = (16, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    for ax, skeleton, label in zip([ax1, ax2, ax3], samples, ['Working', 'Requesting', 'Preparing']):
        mins, maxs = skeleton.min(axis=1), skeleton.max(axis=1)

        num_sk = 10 if norm else 32
        for i in range(num_sk):
            ax.scatter(skeleton[0][i], skeleton[2][i], skeleton[1][i], c='r')
        
        if norm:    
            connections = [[0,1], [1,2], [1,3], [3,4], [4,5], [1,6], [6,7], [7,8], [2,9]]
        else:
            connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11], [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19,20], [20,21], [0,22], [22,23], [23,24], [24,25], [3,26], [26,27], [26, 28], [26,29], [26,30], [26,31]]

        for i, j in connections:
            ax.plot([skeleton[0][i], skeleton[0][j]], [skeleton[2][i], skeleton[2][j]], [skeleton[1][i], skeleton[1][j]], c='r')
            
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[2], maxs[2]])
        ax.set_zlim([mins[1], maxs[1]])
        ax.invert_zaxis()
        ax.set_title(label)
    
    canvas = FigureCanvasTkAgg(fig, 
                               master = right_frame)
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   right_frame) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack()

def plot_paired_skeleton(skeletons, label, norm=True):
    global right_frame

    work1, work2 = label_to_works(int(label))
    label = work1 + "(red) " + work2 + "(blue)"

    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')

    mins, maxs = skeletons.min(axis=-1).min(axis=0), skeletons.max(axis=-1).max(axis=0)

    num_sk = 10 if norm else 32
    for i in range(num_sk):
        ax.scatter(skeletons[0][0][i], skeletons[0][2][i], skeletons[0][1][i], c='r')
        ax.scatter(skeletons[1][0][i], skeletons[1][2][i], skeletons[1][1][i], c='b')
    
    if norm:    
        connections = [[0,1], [1,2], [1,3], [3,4], [4,5], [1,6], [6,7], [7,8], [2,9]]
    else:
        connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11], [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19,20], [20,21], [0,22], [22,23], [23,24], [24,25], [3,26], [26,27], [26, 28], [26,29], [26,30], [26,31]]

    for i, j in connections:
        ax.plot([skeletons[0][0][i], skeletons[0][0][j]], [skeletons[0][2][i], skeletons[0][2][j]], [skeletons[0][1][i], skeletons[0][1][j]], c='r')
        ax.plot([skeletons[1][0][i], skeletons[1][0][j]], [skeletons[1][2][i], skeletons[1][2][j]], [skeletons[1][1][i], skeletons[1][1][j]], c='b')
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    # check
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[2], maxs[2]])
    ax.set_zlim([mins[1], maxs[1]])
    ax.invert_zaxis()
    plt.title(label)
    
    canvas = FigureCanvasTkAgg(fig, 
                               master = right_frame)
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   right_frame) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack()

def plot_both_paired_skeleton(sample_raw, sample_norm, label):
    global right_frame

    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    work1, work2 = label_to_works(int(label))
    label = work1 + "(red) " + work2 + "(blue)"

    fig.suptitle(label)

    for ax, skeleton, norm in zip([ax1, ax2], [sample_raw, sample_norm], [False, True]):
        mins, maxs = skeleton.min(axis=-1).min(axis=0), skeleton.max(axis=-1).max(axis=0)

        num_sk = 10 if norm else 32
        for i in range(num_sk):
            ax.scatter(skeleton[0][0][i], skeleton[0][2][i], skeleton[0][1][i], c='r')
            ax.scatter(skeleton[1][0][i], skeleton[1][2][i], skeleton[1][1][i], c='b')
        
        if norm:    
            connections = [[0,1], [1,2], [1,3], [3,4], [4,5], [1,6], [6,7], [7,8], [2,9]]
        else:
            connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11], [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19,20], [20,21], [0,22], [22,23], [23,24], [24,25], [3,26], [26,27], [26, 28], [26,29], [26,30], [26,31]]

        for i, j in connections:
            ax.plot([skeleton[0][0][i], skeleton[0][0][j]], [skeleton[0][2][i], skeleton[0][2][j]], [skeleton[0][1][i], skeleton[0][1][j]], c='r')
            ax.plot([skeleton[1][0][i], skeleton[1][0][j]], [skeleton[1][2][i], skeleton[1][2][j]], [skeleton[1][1][i], skeleton[1][1][j]], c='b')
    
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[2], maxs[2]])
        ax.set_zlim([mins[1], maxs[1]])
        ax.invert_zaxis()
        ax.set_title("normalized skeleton" if norm else "raw skeleton")
    
    canvas = FigureCanvasTkAgg(fig, 
                               master = right_frame)
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   right_frame) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack()

def animate():
    global frames, right_frame, label, index, animate_id
    
    if not paused:
        if index < len(frames):
            frame = frames[index]
            # index = (index + 1) % len(frames)
            label.configure(image=frame)
            label.image = frame
            update_progress_slider()
            index += 1
        else:
            toggle_pause()
    animate_id = right_frame.after(100, animate)

def toggle_pause():
    global paused, pause_button

    paused = not paused
    if paused:
        pause_button.config(text="Resume")
    else:
        pause_button.config(text="Pause")

def update_progress_slider():
    global progress_slider, index
    progress = (index + 1) / len(frames) * 100
    progress_slider.set(progress)

def jump_to_position(value):
    global index
    position = int(value) / 100 * len(frames)
    index = int(position)
    update_progress_slider()

def get_gif_path():
    global raw_norm, single_paired, grouped_paired, actions, labels

    dataset = single_paired.get() if raw_norm.get() == "raw" else grouped_paired.get()
    label = actions.get() if (dataset == "single") else labels.get()
    idx = np.random.randint(3) if dataset=="single" else 0

    return f"./visualization/gif/{raw_norm.get()}/{dataset}/{label}/{label}_{idx}.gif"

def plot_gif():
    global right_frame, frames, paused, pause_button, label, index, progress_slider, animate_id
    
    right_frame = Frame(window, width=940, height=600, bg='linen')
    right_frame.place(x=440, y=70)
    right_frame.pack_propagate(False)

    gif_path = get_gif_path()
    gif = Image.open(gif_path)

    # split the GIF into frames
    frames = []
    try:
        while True:
            frames.append(ImageTk.PhotoImage(gif.copy()))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    width, height = frames[0].width(), frames[0].height()

    # display the first frame
    label = Label(right_frame, image=frames[0], width=width, height=height)
    label.pack()

    container = Frame(right_frame)
    container.pack()

    # Progress slider
    progress_slider = Scale(container, from_=0, to=100, orient=HORIZONTAL, command=jump_to_position)
    progress_slider.pack(side=LEFT, padx=20)

    # Start animation
    paused = False
    index = 0
    if 'animate_id' in globals():
        right_frame.after_cancel(animate_id)
    a = animate()

    # Pause button
    pause_button = Button(container, text="Pause", command=toggle_pause)
    pause_button.pack(side=LEFT)

def initial_window():
    global img_ani, left_frame, right_frame

    # Create left and right frames
    left_frame = Frame(window, width=400, height=620, bg='linen')
    left_frame.place(x=20, y=60)

    right_frame = Frame(window, width=940, height=600, bg='linen')
    right_frame.place(x=440, y=70)
    right_frame.pack_propagate(False)

    left_label = Label(window, text="Options", font=("Arial", 18), bg="skyblue", fg="black")
    left_label.place(x=170, y=20)

    right_label = Label(window, text="View Skeletons", font=("Arial", 18), bg="skyblue", fg="black")
    right_label.place(x=850, y=30)

    left_box_label = Label(left_frame, text="Choose type of view", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=10)

    # radio button to choose type of view; image or animation
    img_ani = StringVar()
    image_btn = Radiobutton(left_frame, command=img_ani_window ,text="image", value="img", variable=img_ani, font=("Arial", 14), bg='linen')
    animation_btn = Radiobutton(left_frame, command=img_ani_window, text="animation", value="ani", variable=img_ani, font=("Arial", 14), bg='linen')
    image_btn.place(x=10, y=50)
    animation_btn.place(x=140, y=50)

def choose_raw_norm():
    global raw_norm
    if raw_norm.get() == 'raw':
        raw_window()
    
    elif raw_norm.get() == 'norm':
        norm_window()

    elif raw_norm.get() == "recon":
        norm_window()

    else:
        both_window()

def choose_single_paired():
    global img_ani, single_paired, skeleton_dict, label_dict

    if single_paired.get() == "single":
        if img_ani.get() == "img":
            skeleton_dict, _ = load_skeleton_dict()
        single_window()
    
    else:
        if img_ani.get() == "img":
            skeleton_dict, label_dict = load_skeleton_dict()
        paired_window()

def choose_actions():
    global img_ani, raw_norm, actions, skeleton_dict, right_frame

    right_frame = Frame(window, width=940, height=600, bg='linen')
    right_frame.place(x=440, y=70)
    right_frame.pack_propagate(False)

    if img_ani.get() == "img":
        if actions.get() == "working":
            sample = get_single_skeleton(skeleton_dict)
            if raw_norm.get() == "both":
                sample = np.expand_dims(sample, axis=0)
                norm_sample = normalize_skeletons(sample)
                plot_both_single_skeleton(sample.squeeze(), norm_sample.squeeze(), "Working")
            else:
                plot_skeleton(sample, 'Working', norm=False if raw_norm.get()=="raw" else True)
        
        elif actions.get() == 'requesting':
            sample = get_single_skeleton(skeleton_dict)
            if raw_norm.get() == "both":
                sample = np.expand_dims(sample, axis=0)
                norm_sample = normalize_skeletons(sample)
                plot_both_single_skeleton(sample.squeeze(), norm_sample.squeeze(), "Requesting")
            else:
                plot_skeleton(sample, 'Requesting', norm=False if raw_norm.get()=="raw" else True)
        
        elif actions.get() == "preparing":
            sample = get_single_skeleton(skeleton_dict)
            if raw_norm.get() == "both":
                sample = np.expand_dims(sample, axis=0)
                norm_sample = normalize_skeletons(sample)
                plot_both_single_skeleton(sample.squeeze(), norm_sample.squeeze(), "Preparing")
            else:
                plot_skeleton(sample, 'Preparing', norm=False if raw_norm.get()=="raw" else True)

        elif actions.get() == "all":
            samples = get_single_skeleton(skeleton_dict)
            plot_all_skeletons(samples, norm=False if raw_norm.get()=="raw" else True)

    else:
        plot_gif()

def choose_labels():
    global img_ani, raw_norm, labels, skeleton_dict, label_dict, right_frame

    right_frame = Frame(window, width=940, height=600, bg='linen')
    right_frame.place(x=440, y=70)
    right_frame.pack_propagate(False)

    if img_ani.get() == "img":
        if raw_norm.get() == "raw":
            minus_sample, plus_sample = get_paired_skeleton(skeleton_dict, label_dict)
            plot_paired_skeleton(np.array([minus_sample, plus_sample]), labels.get(), norm=False)

        elif raw_norm.get() == "norm":
            sample, _ = get_paired_skeleton(skeleton_dict, label_dict)
            plot_paired_skeleton(np.array([sample[:, :10], sample[:, 10:]]), labels.get(), norm=True)
        
        elif raw_norm.get() == "both":
            minus_sample, plus_sample = get_paired_skeleton(skeleton_dict, label_dict)
            sample_norm = normalize_skeletons(np.array([minus_sample, plus_sample]))
            plot_both_paired_skeleton(np.array([minus_sample, plus_sample]), sample_norm, labels.get())
    
    else:
        plot_gif()

def choose_grouped_paired():
    global img_ani, grouped_paired, skeleton_dict, label_dict

    if grouped_paired.get() == "grouped":
        if img_ani.get() == "img":
            skeleton_dict, _ = load_skeleton_dict()
            single_window()
        else:
            paired_window()
    
    else:
        if img_ani.get() == "img":
            skeleton_dict, label_dict = load_skeleton_dict()
        paired_window()

def choose_both_dataset():
    global img_ani, both_dataset, skeleton_dict, label_dict

    if both_dataset.get() == "grouped":
        if img_ani.get() == "img":
            skeleton_dict = load_skeleton_dict()
        single_window()
    
    else:
        if img_ani.get() == "img":
            skeleton_dict, label_dict = load_skeleton_dict()
        paired_window()

def img_ani_window():
    global img_ani, left_frame, raw_norm

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 50:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of skeleton", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=90)

    # radio button to choose type of skeleton; raw or normalized
    # in the case of animation norm_btn considered as preprocessed
    raw_norm = StringVar()
    raw_btn = Radiobutton(left_frame, command=choose_raw_norm ,text="raw", value="raw", variable=raw_norm, font=("Arial", 14), bg='linen')
    norm_btn = Radiobutton(left_frame, command=choose_raw_norm, text="normalized" if img_ani.get()=="img" else "preprocessed", value="norm", variable=raw_norm, font=("Arial", 14), bg='linen')
    if img_ani.get()=="img":
        both_btn = Radiobutton(left_frame, command=choose_raw_norm ,text="both", value="both", variable=raw_norm, font=("Arial", 14), bg='linen')
        both_btn.place(x=250, y=130)
    else:
        recon_btn = Radiobutton(left_frame, command=choose_raw_norm ,text="reconstruction", value="recon", variable=raw_norm, font=("Arial", 14), bg='linen')
        recon_btn.place(x=250, y=130)
    raw_btn.place(x=10, y=130)
    norm_btn.place(x=100 if img_ani.get()=="img" else 90, y=130)

def ani_window():
    global left_frame

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 50:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of skeleton", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=90)

def raw_window():
    global left_frame, single_paired

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 130:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of dataset", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=170)

    # radio button to choose type of dataset; grouped or paired
    single_paired = StringVar()
    single_btn = Radiobutton(left_frame, command=choose_single_paired ,text="single", value="single", variable=single_paired, font=("Arial", 14), bg='linen')
    paired_btn = Radiobutton(left_frame, command=choose_single_paired, text="paired", value="paired", variable=single_paired, font=("Arial", 14), bg='linen')
    
    single_btn.place(x=10, y=210)
    paired_btn.place(x=140, y=210)

def norm_window():
    global left_frame, grouped_paired

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 130:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of dataset", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=170)

    # radio button to choose type of dataset; grouped or paired
    grouped_paired = StringVar()
    grouped_btn = Radiobutton(left_frame, command=choose_grouped_paired ,text="single", value="grouped", variable=grouped_paired, font=("Arial", 14), bg='linen')
    paired_btn = Radiobutton(left_frame, command=choose_grouped_paired, text="paired", value="paired", variable=grouped_paired, font=("Arial", 14), bg='linen')
    
    grouped_btn.place(x=10, y=210)
    paired_btn.place(x=140, y=210)

def both_window():
    global left_frame, both_dataset

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 130:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of dataset", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=170)

    # radio button to choose type of dataset; grouped or paired
    both_dataset = StringVar()
    grouped_btn = Radiobutton(left_frame, command=choose_both_dataset ,text="single", value="grouped", variable=both_dataset, font=("Arial", 14), bg='linen')
    paired_btn = Radiobutton(left_frame, command=choose_both_dataset, text="paired", value="paired", variable=both_dataset, font=("Arial", 14), bg='linen')
    
    grouped_btn.place(x=10, y=210)
    paired_btn.place(x=140, y=210)

def single_window():
    global left_frame, raw_norm, actions

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 210:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of dataset", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=250)

    # radio button to choose type of actions; working, requesting, preparing
    actions = StringVar()
    wokring_btn = Radiobutton(left_frame, command=choose_actions ,text="working", value="working", variable=actions, font=("Arial", 14), bg='linen')
    requesting_btn = Radiobutton(left_frame, command=choose_actions, text="requesting", value="requesting", variable=actions, font=("Arial", 14), bg='linen')
    preparing_btn = Radiobutton(left_frame, command=choose_actions, text="preparing", value="preparing", variable=actions, font=("Arial", 14), bg='linen')
    if raw_norm.get() != "both":
        all_btn = Radiobutton(left_frame, command=choose_actions, text="all", value="all", variable=actions, font=("Arial", 14), bg='linen')

    wokring_btn.place(x=10, y=290)
    requesting_btn.place(x=130, y=290)
    preparing_btn.place(x=270, y=290)
    if raw_norm.get() != "both":
        all_btn.place(x=10, y=330)

def paired_window():
    global left_frame, labels

    # reset frame
    for widget in left_frame.winfo_children():
        if int(widget.place_info()['y']) > 210:
            widget.destroy()

    left_box_label = Label(left_frame, text="Choose type of dataset", font=("Arial", 16), bg="linen", fg="black")
    left_box_label.place(x=10, y=250)

    # radio button to choose type of labels/class; w(working), r(requesting), p(preparing)
    labels = StringVar()
    ww_btn = Radiobutton(left_frame, command=choose_labels ,text="ww", value=1, variable=labels, font=("Arial", 14), bg='linen')
    wr_btn = Radiobutton(left_frame, command=choose_labels, text="wr", value=2, variable=labels, font=("Arial", 14), bg='linen')
    wp_btn = Radiobutton(left_frame, command=choose_labels, text="wp", value=3, variable=labels, font=("Arial", 14), bg='linen')
    rw_btn = Radiobutton(left_frame, command=choose_labels ,text="rw", value=4, variable=labels, font=("Arial", 14), bg='linen')
    rr_btn = Radiobutton(left_frame, command=choose_labels, text="rr", value=5, variable=labels, font=("Arial", 14), bg='linen')
    rp_btn = Radiobutton(left_frame, command=choose_labels, text="rp", value=6, variable=labels, font=("Arial", 14), bg='linen')
    pw_btn = Radiobutton(left_frame, command=choose_labels ,text="pw", value=7, variable=labels, font=("Arial", 14), bg='linen')
    pr_btn = Radiobutton(left_frame, command=choose_labels, text="pr", value=8, variable=labels, font=("Arial", 14), bg='linen')
    pp_btn = Radiobutton(left_frame, command=choose_labels, text="pp", value=9, variable=labels, font=("Arial", 14), bg='linen')

    ww_btn.place(x=10, y=290)
    wr_btn.place(x=130, y=290)
    wp_btn.place(x=270, y=290)
    rw_btn.place(x=10, y=330)
    rr_btn.place(x=130, y=330)
    rp_btn.place(x=270, y=330)
    pw_btn.place(x=10, y=370)
    pr_btn.place(x=130, y=370)
    pp_btn.place(x=270, y=370)

# the main Tkinter window 
window = Tk() 
  
# setting the title  
window.title('Visualize Skeleton') 
  
# dimensions of the main window 
window.geometry("1400x700")
window.config(bg="skyblue")

# initialize window
initial_window()

# run the gui 
window.mainloop()