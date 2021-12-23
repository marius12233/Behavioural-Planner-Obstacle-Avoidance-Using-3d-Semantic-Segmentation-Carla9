#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
from copy import deepcopy

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append("../")
sys.path.append("../carla")

sys.path.append("SalsaNext/")
sys.path.append("SalsaNext/train")
sys.path.append("SalsaNext/salsanext_64")
from infer_test import load

import carla

import random
import time
from queue import Queue
from queue import Empty
import open3d as o3d
import numpy as np
import cv2
import math

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import LocalPlanner
import behavioural_planner
from matplotlib import cm
from world_projector import ToWorldProjector, transform_world_to_ego_frame

###############################################################################

# ABILITAZIONE PRINT DI DEBUG
DRAW_WAYPOINS = False
PRINT_OBSTACLE = True
PRINT_VELOCITY = False


# DIMENSIONI DELLA CAMERA
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 512

# COSTANTI PER LA NAVIGAZIONE (1: gt_segmentation, 2: SalsaNext)
NAVIGATION_MODE = 1


# SEGMENTATION PARAMETERS
MIDDLE_LANE = 6

if NAVIGATION_MODE == 1:
    ROAD = 7
    OTHER = 0
    VEHICLE = 10
    PEDESTRIAN = 4

elif NAVIGATION_MODE == 2:
    ROAD = 40
    SIDEWALK = 48
    OTHER = 255
    VEHICLE = 10
    PEDESTRIAN = 30

OBSTACLES_CARLA_LIST = [4, 10]  # 4 pedestrian, 10 vehicle
SHOW_SEG_GT = 1
MIN_FRACTION_AREA_OBSTACLE = 0.00031  # 100 pixels in a 800*400 image

###################################################################
#####################MANAGEMENT OF WORLD###########################
###################################################################
NUMBER_OF_PEDESTRIAN = 10#200
NUMBER_OF_VEHICLE = 15
START_INDEX = 88  # 88 5
END_INDEX = 133  #133

FPS = 0.1  # 0.1 =10fps
DESIRED_SPEED = 18
TURN_SPEED = 10
APPROACHING_TF_SPEED = 14
LOOKAHEAD = 20
VEHICLE_LOOKAHEAD = 16
LP_FREQUENCY_DIVISOR = 2

KP = 0.58
KD = 0.4
KI = 0.5
DT = 0.15
###################################################################
###################################################################
###################################################################

###################################################################
##################MANAGEMENT  OF LIDARS ###############################
###################################################################
lidar_parameters = {}
LIDAR = 16
RANGE_ANGLE_MIN = 45
RANGE_ANGLE_MAX = 135
X_MAX_ROAD = 10
X_MIN_ROAD = 5
X_MAX_VEHICLE = 30
X_MIN_VEHICLE = 15
X_OFFSET_VEHICLE = 2.5
X_MAX_PEDESTRIAN = 20
X_MAX_PEDESTRIAN_TURN = 10
X_MIN_PEDESTRIAN = 10

MAX_VEHICLE_COLOR = [0.0, 1.0, 0.0]
MIN_VEHICLE_COLOR = [0.0, 1.0, 0.0]
MAX_PEDESTRIAN_COLOR = [0.0, 0.0, 1.0]
MIN_PEDESTRIAN_COLOR = [0.0, 0.0, 1.0]
LEFT_ROADWAY_COLOR = [1.0, 1.0, 0.4]
RIGHT_ROADWAY_COLOR = [1.0, 1.0, 0.4]
COLOR_HEIGHT = 0.1

Y_STD_MIN = -6 #-3
Y_STD_MAX = 6 #4

LANE_REDUCTION_FACTOR = 3


###################################################################
###################################################################
###################################################################
# Camera parameters
camera_parameters = {}
camera_parameters['x'] = 1.73
camera_parameters['y'] = 0
camera_parameters['z'] = 1.3
camera_parameters['width'] = CAMERA_WIDTH
camera_parameters['height'] = CAMERA_HEIGHT
camera_parameters['fov'] = 90
camera_parameters['yaw'] = 0
camera_parameters['pitch'] = 0
camera_parameters['roll'] = 0



projector = ToWorldProjector(camera_parameters)
###############################################################################

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (70, 70, 70),  # (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # normalize each channel [0-1] since is what Open3D uses

def kitti_to_cityscapes_tag(np_array_tag):
    np_array_tag=np.array(np_array_tag)
    new_array_tag=np.zeros_like(np_array_tag)
    length=np_array_tag.shape[0]

    for j  in range(0,length):
        if (np_array_tag[j]== 0):
            new_array_tag[j]= 0
        if(np_array_tag[j]==50):
            new_array_tag[j]=1
        if (np_array_tag[j]== 51):
            new_array_tag[j] = 2
        if (np_array_tag[j]== 0):
            new_array_tag[j] = 3
        if (np_array_tag[j] == 30):
            new_array_tag[j] = 4
        if (np_array_tag[j]== 80):
            new_array_tag[j]= 5
        if (np_array_tag[j]== 60):
            new_array_tag[j]= 6
        if (np_array_tag[j]== 40):
            new_array_tag[j]= 7
        if (np_array_tag[j]== 48):
            new_array_tag[j]= 8
        if (np_array_tag[j]== 70):
            new_array_tag[j]= 9
        if (np_array_tag[j]== 10):
            new_array_tag[j] = 10
        if (np_array_tag[j]== 52):
            new_array_tag[j]= 11
        if (np_array_tag[j]== 81):
            new_array_tag[j]= 12
        if (np_array_tag[j]== 0):
            new_array_tag[j]= 13
        if (np_array_tag[j]== 49):
            new_array_tag[j]= 14
        if (np_array_tag[j] == 52):
            new_array_tag[j] = 15
        if (np_array_tag[j] == 16):
            new_array_tag[j] = 16
        if (np_array_tag[j] == 51):
            new_array_tag[j] = 17
        if (np_array_tag[j]== 81):
            new_array_tag[j] = 18
        if (np_array_tag[j]== 99):
            new_array_tag[j] = 19
        if (np_array_tag[j]== 0):
            new_array_tag[j] = 20
        if (np_array_tag[j] == 0):
            new_array_tag[j] = 21
        if (np_array_tag[j]== 72):
            new_array_tag[j] = 22
    #print('VECCHIO ARRAY:',np_array_tag)
    #print('############################################################')
    #print('############################################################')
    #print('############################################################')
    #print('NUOVO ARRAY:', new_array_tag)
    
    return new_array_tag


SIDEWALK = 8
USE_SIDEWALK = False

def get_label_mask(color_mask):
    w = color_mask.shape[0]
    h = color_mask.shape[1]
    label_mask = np.zeros(shape=[w, h, 1], dtype=np.uint8)
    unique_labels = set(tuple(v) for m2d in color_mask for v in m2d)

    # unique_labels = [v for v in np.unique(color_mask[:,:,:3])]
    print("Unique labels: ", unique_labels)
    for label in unique_labels:
        print("Label: ", label)
        idx = np.where(color_mask == label)
        print("idx: ", idx[:2])
        label_id = np.where(LABEL_COLORS == label)
        label_mask[idx[:2]] = label_id

    return label_mask


def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)


def depth_to_array(array):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def send_control_command(player, throttle, steer, brake,
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.

    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = carla.VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    player.apply_control(control)


def convert_waypoints_format(waypoints):
    """[summary]

    Args:
        waypoints ([List]): [List of waypoints to convert in our format: (x, y, v)]

    Returns:
        [List]: [List of waypoints in our format ]
    """
    waypoint_list = []

    for w in waypoints:
        location = w[0].transform.location
        x, y = location.x, location.y
        v = DESIRED_SPEED
        if w[0].is_junction:
            v = TURN_SPEED

        waypoint_list.append([x, y, v])

    return waypoint_list


def get_current_pose(vehicle):
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    ego_x, ego_y, ego_z = location.x, location.y, location.z
    ego_roll = math.radians(rotation.roll)
    ego_yaw = math.radians(rotation.yaw)
    ego_pitch = math.radians(rotation.pitch)
    return ego_x, ego_y, ego_z, ego_pitch, ego_roll, ego_yaw


def get_speed(vehicle):
    """
    Calcolo velocità veioli in km orari
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def semantic_lidar_callback(point_cloud, point_list, labels_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels: np.array = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    labels_list.clear()
    labels_list += labels.tolist()



def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def filter_pcd(pcd, point_list, labels_list: list, x_max_pedestrian_custom,x_max_vehicle_custom):
    pedestrian_ahead = False
    vehicle_far_ahead = False
    vehicle_near_ahead = False
    # print("Len labels list: ", len(labels_list))

    lidar_data = np.asarray(point_list.points)
    if len(lidar_data) == 0:
        print("Lidar data is 0")
        return
    lidar_colors = np.asarray(point_list.colors)
    lidar_semantic = np.asarray(labels_list)
    print("Lidar data shape:: ", lidar_data.shape)
    print("Lidar color shape: ", lidar_colors.shape)
    print("Lidar semantic shape: ", lidar_semantic.shape)
    points = []
    point_colors = []
    y_road_points = []
    y_sidewalk_points = []
    pedestrian_points = []
    vehicle_points = []

    for i in range(lidar_data.shape[0]):
        x = lidar_data[i][0]
        y = lidar_data[i][1]
        z = lidar_data[i][2]
        if x > 0:
            angle = np.arctan2(x, y) * 180 / np.pi
            if angle > RANGE_ANGLE_MIN and angle < RANGE_ANGLE_MAX:
                points.append([x, y, z])
                point_colors.append([lidar_colors[i][0], lidar_colors[i][1], lidar_colors[i][2]])
                if lidar_semantic[
                    i] == ROAD and x < X_MAX_ROAD and x > X_MIN_ROAD and y > Y_STD_MIN and y < Y_STD_MAX:  # Prendiamo una piccola regione davanti la macchina per trovare le linee di fine strada
                    y_road_points.append(y)
                if lidar_semantic[
                    i] == SIDEWALK and x < X_MAX_ROAD and x > X_MIN_ROAD and y<0:  # Prendiamo una piccola regione davanti la macchina a destra per trovare i marciapiedi in curva e settare quelli come linee di fine strada
                    y_sidewalk_points.append(y)
                if lidar_semantic[i] == PEDESTRIAN:
                    pedestrian_points.append([x, y, z])
                if lidar_semantic[i] == VEHICLE:
                    vehicle_points.append([x, y, z])

    if len(y_road_points) > 0:
        # print("list: ", y_road_points)

        y_min = min(y_road_points)
        y_max = max(y_road_points)
        
        #Trying to solve problem of pedestrian in turns
        y_min_sidewalk = -np.Inf
        if len(y_sidewalk_points)  and USE_SIDEWALK> 0:
            y_min_sidewalk =min(y_sidewalk_points)
            
        y_min = max(y_min, y_min_sidewalk)
        if(PRINT_OBSTACLE):
            print()
            print("ymin found: ", y_min)
            print("ymax found: ", y_max)

        ########## PEDESTRIAN DETECTION ##########
        filtered_pedestrian_points = []
        for pedestrian_point in pedestrian_points:
            x_p, y_p, z_p = pedestrian_point
            if y_p >= y_min and y_p <= y_max and x_p <= x_max_pedestrian_custom:
                filtered_pedestrian_points.append(pedestrian_point)
        if len(filtered_pedestrian_points) > 0:
            filtered_pedestrian_points = np.array(filtered_pedestrian_points)
            idx_pedestrian_of_interest = np.argmin(filtered_pedestrian_points[:,
                                                   0])  # Il punto più vicino a noi del pedone per il quale ci stiamo fermando
            pedestrian_point_of_interest = filtered_pedestrian_points[idx_pedestrian_of_interest]
            pedestrian_ahead = True
            if (PRINT_OBSTACLE):
                print("########## Pedestrian point of interest: ", pedestrian_point_of_interest)
        else:
            if (PRINT_OBSTACLE):
                print("No Pedestrian found !!!")

        ##########  NEAR VEHICLE DETECTION ##########
        filtered_vehicle_points_near = []
        for vehicle_point in vehicle_points:
            x_v, y_v, z_v = vehicle_point
            if y_v >= y_min and y_v <= y_max / LANE_REDUCTION_FACTOR and x_v <= X_MIN_VEHICLE and x_v >= X_OFFSET_VEHICLE:
                filtered_vehicle_points_near.append(vehicle_point)
        if len(filtered_vehicle_points_near) > 0:
            filtered_vehicle_points_near = np.array(filtered_vehicle_points_near)
            idx_vehicle_of_interest_near = np.argmin(filtered_vehicle_points_near[:,
                                                     0])  # Il punto più vicino a noi del pedone per il quale ci stiamo fermando
            near_vehicle_point_of_interest = filtered_vehicle_points_near[idx_vehicle_of_interest_near]
            vehicle_near_ahead = True
            if (PRINT_OBSTACLE):
                print("########## NEAR Vehicle point of interest: ", near_vehicle_point_of_interest)
        else:
            if (PRINT_OBSTACLE):
                print("No NEAR Vehicle found!!!")

        ##########  FAR VEHICLE DETECTION ##########
        filtered_vehicle_points_far = []
        for vehicle_point in vehicle_points:
            x_v, y_v, z_v = vehicle_point
            if y_v >= y_min and y_v <= y_max / LANE_REDUCTION_FACTOR and x_v <= x_max_vehicle_custom and x_v > X_MIN_VEHICLE:
                filtered_vehicle_points_far.append(vehicle_point)
        if len(filtered_vehicle_points_far) > 0:
            filtered_vehicle_points_far = np.array(filtered_vehicle_points_far)
            idx_vehicle_of_interest_far = np.argmin(filtered_vehicle_points_far[:,
                                                    0])  # Il punto più vicino a noi del pedone per il quale ci stiamo fermando
            far_vehicle_point_of_interest = filtered_vehicle_points_far[idx_vehicle_of_interest_far]
            vehicle_far_ahead = True
            if (PRINT_OBSTACLE):
                print("########## FAR Vehicle point of interest: ", far_vehicle_point_of_interest)
        else:
            if (PRINT_OBSTACLE):
                print("No FAR Vehicle found!!!")

    geometry_visualization(points, y_min, y_max, point_colors, pcd, x_max_pedestrian_custom,x_max_vehicle_custom)
    return pedestrian_ahead, vehicle_far_ahead, vehicle_near_ahead


def geometry_visualization(points, y_min, y_max, point_colors, pcd, x_max_pedestrian_custom,x_max_vehicle_custom):
    if len(points) > 0:
        y_m = -5
        y_mm = 5
        for y in range(y_m, y_mm):
            if (y >= y_min and y <= y_max / 3):
                points.append([float(x_max_vehicle_custom), float(y), COLOR_HEIGHT])
                points.append([float(X_MIN_VEHICLE), float(y), COLOR_HEIGHT])
                point_colors.append(MAX_VEHICLE_COLOR)
                point_colors.append(MIN_VEHICLE_COLOR)
            points.append([float(X_MIN_PEDESTRIAN), float(y), COLOR_HEIGHT])
            points.append([float(x_max_pedestrian_custom), float(y), COLOR_HEIGHT])
            point_colors.append(MIN_PEDESTRIAN_COLOR)
            point_colors.append(MAX_PEDESTRIAN_COLOR)

        x_m = 10
        x_mm = 20
        for x in range(x_m, x_mm):
            points.append([x, y_max, COLOR_HEIGHT])
            point_colors.append(LEFT_ROADWAY_COLOR)
            points.append([x, y_min, COLOR_HEIGHT])
            point_colors.append(RIGHT_ROADWAY_COLOR)

    sample_pts = np.array(points)
    sample_colors = np.array(point_colors)
    if len(sample_pts) > 0:
        pcd.points = o3d.open3d.utility.Vector3dVector(sample_pts)
        pcd.colors = o3d.open3d.utility.Vector3dVector(sample_colors)
    else:
        print("Sample pts is 0")
    # return pcd


def set_pid_for_velocity():
    if (DESIRED_SPEED == 18):
        KP = 0.58
        KD = 0.4
        KI = 0.5
        DT = 0.15
    elif (DESIRED_SPEED == 28):
        KP = 0.58
        KD = 0.4
        KI = 0.4
        DT = 0.03
    elif (DESIRED_SPEED == 43):
        KP = 0.18
        KD = 0.50
        KI = 0.60
        DT = 0.15


def set_lidar_parameters():
    if (LIDAR == 16):
        lidar_parameters['x'] = 0
        lidar_parameters['y'] = 0
        lidar_parameters['z'] = 1.73  # 1.8+1.73
        lidar_parameters['yaw'] = 0
        lidar_parameters['pitch'] = 0
        lidar_parameters['roll'] = 0
        lidar_parameters['upper_fov'] = 15
        lidar_parameters['lower_fov'] = -15
        lidar_parameters['range'] = 100
        lidar_parameters['rotation_frequency'] = 10
        lidar_parameters['points_per_second'] = 300000
        lidar_parameters['channels'] = 16
    elif (LIDAR == 64):
        lidar_parameters['x'] = -0.5
        lidar_parameters['y'] = 0
        lidar_parameters['z'] = 3.53  # 1.8+1.73
        lidar_parameters['yaw'] = 0
        lidar_parameters['pitch'] = 0
        lidar_parameters['roll'] = 0
        lidar_parameters['upper_fov'] = 3
        lidar_parameters['lower_fov'] = -25
        lidar_parameters['range'] = 120
        lidar_parameters['rotation_frequency'] = 10
        lidar_parameters['points_per_second'] = 1300000
        lidar_parameters['channels'] = 64
    else:
        print("SELEZIONARE SOLO LIDAR A 64 O 16 BEAM!!!")

def predict_model(point_list, user_runtime):
    predicting_data = np.asarray(point_list.points)
    predicting_points = predicting_data.reshape((-1, 3))
    print("Shape points: ", predicting_points.shape)
    reflectance_array = np.ones((predicting_points.shape[0],1))
    print("Shape reflectance: ", reflectance_array.shape)
    predicting_np_points = np.concatenate((predicting_points, reflectance_array), axis=-1)
    pred_labels = user_runtime.infer(predicting_np_points)
    
    return pred_labels

def main():
    actor_list = []
    set_pid_for_velocity()
    set_lidar_parameters()

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        # Once we have a client we can retrieve the world that we want
        world = client.load_world("Town01")
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()
        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = FPS
        settings.synchronous_mode = True
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        ################### Initialize  MISSION PLANNER COPY ###################
        map = world.get_map()
        sampling_resolution = 1

        dao = GlobalRoutePlannerDAO(map, sampling_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()


        spawn_points = world.get_map().get_spawn_points()
        a = carla.Location(spawn_points[START_INDEX].location)
        print("Positions: ", a.x, a.y)
        b = carla.Location(spawn_points[END_INDEX].location)
        w1 = grp.trace_route(a, b)

        waypoints = convert_waypoints_format(w1)
        if DRAW_WAYPOINS:
            for w in w1:
                mark = w[0].lane_id

                world.debug.draw_string(w[0].transform.location, str(mark), draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1000,
                                        persistent_lines=True)

                world.debug.draw_point(w[0].transform.location, 0.1,
                                       color=carla.Color(r=255, g=0, b=0), life_time=0,
                                       persistent_lines=True)

        ###################################################################

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = blueprint_library.filter('model3')[0]

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = world.get_map().get_spawn_points()[START_INDEX]

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        ## Initialize local planner with controller
        args_lateral_dict = {
            'K_P': KP,  # 0.2
            'K_D': KD,  # 0.05
            'K_I': KI,  # 0.1
            'dt':  DT}  # 1.0/20.0

        local_planner = LocalPlanner(
            vehicle, opt_dict={'target_speed': DESIRED_SPEED,
                               'lateral_control_dict': args_lateral_dict})

        route = grp.trace_route(
            a,
            b)

        local_planner.set_global_plan(route)

        # vehicle.set_location(a)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(False)

        ### PEDESTRIAN
        for _ in range(0, NUMBER_OF_PEDESTRIAN):
            # temp_transform = random.choice(transform)
            # temp_transform.location.x+=20
            # temp_transform.location.y += 1.5
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc

            bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if bp.has_attribute('speed'):
                direction = carla.Vector3D(round(random.uniform(0.6, 1.8), 1), round(random.uniform(0, 0.5), 1), 0)
                # direction = carla.Vector3D(0.9,0.1,0)
                # bp.apply_control(arla.WalkerControl(direction,1.7,False))
                # carla.WalkerControl(direction,1.7,False)
            npc = world.try_spawn_actor(bp, spawn_point)

            if npc is not None:
                actor_list.append(npc)
                # traffic_manager.collision_detection(vehicle, npc, False)
                npc.apply_control(carla.WalkerControl(direction, 1.0, False))
                print('created %s' % npc.type_id)

        ####### /PEDESTRIAN

        # Let's add now a RGB camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')  # [0]
        camera_bp.set_attribute("image_size_x", str(camera_parameters['width']))
        camera_bp.set_attribute("image_size_y", str(camera_parameters['height']))
        camera_bp.set_attribute("fov", str(camera_parameters["fov"]))
        camera_transform = carla.Transform(carla.Location(x=camera_parameters["x"], z=camera_parameters["z"]))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        # Let's add now a SEMANTIC LIDAR  camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('upper_fov', str(lidar_parameters['upper_fov']))
        lidar_bp.set_attribute('lower_fov', str(lidar_parameters['lower_fov']))
        lidar_bp.set_attribute('channels', str(lidar_parameters['channels']))
        lidar_bp.set_attribute('range', str(lidar_parameters['range']))
        lidar_bp.set_attribute('rotation_frequency', str(lidar_parameters['rotation_frequency']))
        lidar_bp.set_attribute('points_per_second', str(lidar_parameters['points_per_second']))
        lidar_transform = carla.Transform(carla.Location(x=lidar_parameters['x'], z=lidar_parameters['z']))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # Oh wait, I don't like the location we gave to the vehicle, I'm going
        # to move it a bit forward.
        location = vehicle.get_location()
        # location.x += 40
        # vehicle.set_location(location)
        print('moved vehicle to %s' % location)

        # But the city now is probably quite empty, let's add a few more
        # vehicles.
        # transform.location += carla.Location(x=40, y=-3.2)
        # transform.rotation.yaw = -180.0

        for _ in range(0, NUMBER_OF_VEHICLE):
            transform_vehicles = world.get_map().get_spawn_points()
            temp_transform = random.choice(transform_vehicles)
            sbp = random.choice(blueprint_library.filter('vehicle'))
            print
            while (str(sbp.id) == "vehicle.harley-davidson.low_rider" or str(sbp.id) == "vehicle.kawasaki.ninja"
                   or str(sbp.id) == "vehicle.carlamotors.carlacola" or str(sbp.id) == "vehicle.bh.crossbike"
                   or str(sbp.id) == "vehicle.yamaha.yzf"):
                sbp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(sbp, temp_transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot()
                print('created %s' % npc.type_id)

        # Compute the list of traffic lights
        actor_list = world.get_actors()
        tf_list = actor_list.filter("*traffic_light*")

        traffic_light_list = []

        for tf in tf_list:
            traffic_light_list.append(tf)

        ## BEHAVIOURAL PLANNER CREATIOIN
        bp = behavioural_planner.BehaviouralPlanner(LOOKAHEAD, VEHICLE_LOOKAHEAD, None, map, vehicle,
                                                    traffic_light_list)

        image_queue = Queue()
        point_list = o3d.geometry.PointCloud()
        labels_list = []

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: semantic_lidar_callback(data, point_list, labels_list))
        
        #Initialize model
        model_dir = "SalsaNext/salsanext_16"
        user_runtime = load(model_dir)
        

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True
        pcd = o3d.open3d.geometry.PointCloud()
        first_lidar = False
        pedestrian_ahead = False
        vehicle_far_ahead = False
        vehicle_near_ahead = False
        vehicle_turning = False

        for frame in range(10000000):
            world.tick()
            world_frame = world.get_snapshot().frame

            current_x, current_y, current_z, current_pitch, current_roll, current_yaw = \
                get_current_pose(vehicle)

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue


            current_pos = [current_x, current_y, current_z]
            current_rot = [current_roll, current_pitch, current_yaw]

            if frame % LP_FREQUENCY_DIVISOR == 0:

                # Show RGB camera
                # Get the raw BGRA buffer and convert it to an array of RGB of
                # shape (image_data.height, image_data.width, 3).
                im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
                im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
                im_array = im_array[:, :, :3][:, :, ::-1]
                im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                interest_data = np.copy(im_array)
                cv2.imshow("CAM RGB", im_array)

                # Behavioural planner transition

                current_speed = get_speed(vehicle)
                ego_state = [current_x, current_y, current_yaw, current_speed]

                bp._front_pedestrian = pedestrian_ahead
                bp._front_obstacle = vehicle_near_ahead
                bp._lead_vehicle = vehicle_far_ahead
                #bp.transition_state(waypoints, ego_state, current_speed)

                if (bp._state == 5):
                    vehicle_turning = True
                    bp.transition_state(waypoints, ego_state, current_speed)
                    local_planner.set_speed(TURN_SPEED)
                else:
                    vehicle_turning = False
                    if (bp._state == 3):
                        bp.transition_state(waypoints, ego_state, current_speed)
                        local_planner.set_speed(APPROACHING_TF_SPEED)

                    elif (bp._state == 1 or bp._state == 2):
                        vehicle_turning = True
                        bp.transition_state(waypoints, ego_state, current_speed)
                        local_planner.set_speed(0)
                    else:
                        bp.transition_state(waypoints, ego_state, current_speed)
                        local_planner.set_speed(DESIRED_SPEED)

                if len(point_list.points) > 0 and first_lidar is False:
                    first_lidar = True
                    point_list_cpy = deepcopy(point_list)
                    pred_labels = predict_model(point_list_cpy, user_runtime)
                    pred_labels = kitti_to_cityscapes_tag(pred_labels[0])
                    int_color = LABEL_COLORS[pred_labels]
                    point_list_cpy.colors = o3d.utility.Vector3dVector(int_color)
                    
                    print("Shape pred label: ", len(pred_labels))
                    print("Labels list: ", len(labels_list))
                    
                    if (vehicle_turning is False):
                        #Prediction
                        
                        #print("predicted labels: ", pred_labels)
                        pedestrian_ahead, vehicle_far_ahead, vehicle_near_ahead = filter_pcd(pcd, point_list_cpy,
                                                                                             pred_labels.tolist(),
                                                                                             X_MAX_PEDESTRIAN,X_MAX_VEHICLE)
                    else:
                        
                        #print("predicted labels: ", pred_labels)
                        pedestrian_ahead, vehicle_far_ahead, vehicle_near_ahead = filter_pcd(pcd, point_list_cpy,
                                                                                             pred_labels.tolist(),
                                                                                             X_MAX_PEDESTRIAN_TURN,X_MIN_VEHICLE)
                        
                    
                    vis.add_geometry(pcd)

                # TRASFORMATION######################
                if len(point_list.points) > 0 and first_lidar:
                    point_list_cpy = deepcopy(point_list)
                    
                    print("Shape points in: ", len(point_list.points))
                    pred_labels = predict_model(point_list_cpy, user_runtime)
                    print("Shape pred label before: ", len(pred_labels[0]))
                    pred_labels = kitti_to_cityscapes_tag(pred_labels[0])
                    int_color = LABEL_COLORS[pred_labels]
                    point_list_cpy.colors = o3d.utility.Vector3dVector(int_color)
                    print("Shape pred label: ", len(pred_labels))
                    print("Labels list: ", len(labels_list))
                    
                    if (vehicle_turning is False):
                        #print("predicted labels: ", pred_labels)
                        pedestrian_ahead, vehicle_far_ahead, vehicle_near_ahead = filter_pcd(pcd, point_list_cpy,
                                                                                             pred_labels.tolist(),
                                                                                             X_MAX_PEDESTRIAN,X_MAX_VEHICLE)
                    else:
                        #print("predicted labels: ", pred_labels)
                        pedestrian_ahead, vehicle_far_ahead, vehicle_near_ahead = filter_pcd(pcd, point_list_cpy,
                                                                                             pred_labels.tolist(),
                                                                                             X_MAX_PEDESTRIAN_TURN,X_MIN_VEHICLE)
                   
                
                vis.update_geometry(pcd)
                # vis.update_geometry(point_list)
                vis.poll_events()
                vis.update_renderer()
                # # This can fix Open3D jittering issues:
                time.sleep(0.05)
                if (PRINT_VELOCITY):
                    print('ACTUAL VELOCITY : ', str(get_speed(vehicle)))

                control = local_planner.run_step(debug=False)
                # print("Control: ", control)

                send_control_command(vehicle, control.throttle, control.steer, control.brake)


    finally:

        print('destroying actors')
        traffic_manager.set_synchronous_mode(False)
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    main()
