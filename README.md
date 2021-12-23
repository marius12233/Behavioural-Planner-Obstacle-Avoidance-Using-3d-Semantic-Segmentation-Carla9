# Behavioural-Planner-Obstacle-Avoidance-Using-3d-Semantic-Segmentation-Carla9
Behavioural Planner for Obstacle Avoidance on Carla 9 based on 3D Semantic Segmentation of Point Cloud using SalsaNext

This repo is the first project I did for my thesis work.
It is an Obstacle Avoidance algorithm on Carla using information given by 3D Semantic Segmentation of Point Cloud.
Both the Perception Module (included in the main) and a Behavioural Planner were implemented from scratch.
The net used in this project is SalsaNext.
The global planner and local planner used are those provided by Carla Simulator.

Instructions for run:

1. Download and extract folder SalsaNext from here: (https://drive.google.com/drive/folders/1M4ru3266Ukc9oluf0z8MFxAu6yxnmUTn?usp=sharing). This contains configuration files and weights of Salsanext trained on both 16-beams lidar and 64-beams lidar on synthetic dataset of Pointclouds from Carla.
2. Run Carla Simulator Server on localhost and port 2000.
3. Run `python main_prediction.py` to start a simulation which uses SalsaNext trained on 64 beams.

The output is composed by 2 windows, one for the camera and the other for the Lidar.

The windows should be like these:

![Output screen](https://github.com/marius12233/Behavioural-Planner-Obstacle-Avoidance-Using-3d-Semantic-Segmentation-Carla9/blob/main/images/Untitled%20Diagram.drawio.png)

