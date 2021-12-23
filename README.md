# Behavioural-Planner-Obstacle-Avoidance-Using-3d-Semantic-Segmentation-Carla9
Behavioural Planner for Obstacle Avoidance on Carla 9 based on 3D Semantic Segmentation of Point Cloud using SalsaNext

This repo is the first project I did for my thesis work.
It is an Obstacle Avoidance algorithm on Carla using information given by 3D Semantic Segmentation of Point Cloud.
Both the Perception Module (included in the main) and a Behavioural Planner were implemented from scratch.
The net used in this project is SalsaNext.
The global planner and local planner used are those provided by Carla Simulator.

Instructions for run:

1. Run Carla Simulator Server on localhost and port 2000.
2. Run `python main_prediction.py` to start a simulation which uses SalsaNext trained on 64 beams.

The output is composed by 2 windows, one for the camera and the other for the Lidar.

The windows should be like these:

![Output screen](https://github.com/marius12233/Behavioural-Planner-Obstacle-Avoidance-Using-3d-Semantic-Segmentation-Carla9/blob/main/images/Untitled%20Diagram.drawio.png)

