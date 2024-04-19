# Multi Users Activities Recognition for Human-Robot Collaboration

This project explores the challenge of gathering data for multi-user interactions in Human-Robot collaboration. By merging the data collected from individual users to produce multi users data, the study aims to simplify the process of dataset creation. Using 3D skeleton poses of activities performed by single users, the project demonstrates the feasibility of training machine learning models, such as LSTM networks and VAEs with STGCNs, to recognize activities of user pairs. The results indicate that this approach achieves comparable performance to training data collected from groups of individual users, offering a promising solution for advancing research in multi-party Human-Robot interaction and collaboration.

## Overview

* [config/](config) includes json files saving the model hyperparameter settings
* [csf/](csf) includes codes to control the CSF3
* [data/](data) includes skeleton data
* [preprocess/](preprocess) includes code to preprocess data
* [visualization/](visualization) includes code for visualizing the data and some visualization

## Data
<img align="left" width="400" height="400" src="./visualization/gif/raw/paired/1/1_0.gif">
|First Image|Second Image|
|:-:|:-:|
|![First Image](./visualization/gif/raw/paired/1/1_0.gif)|![Second Image](./visualization/gif/raw/paired/1/1_0.gif)|
