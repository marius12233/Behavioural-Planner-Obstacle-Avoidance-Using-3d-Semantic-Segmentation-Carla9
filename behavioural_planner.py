#!/usr/bin/env python3
import numpy as np
import math
import cv2
from math import pi
from tools.misc import is_within_distance_ahead
import carla
from main import NAVIGATION_MODE, MIDDLE_LANE, ROAD, OTHER, VEHICLE, PEDESTRIAN
import transforms3d

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
APPROACHING_SEMAPHORE = 3
LEAVING_SEMAPHORE = 5

# Stop speed threshold
STOP_THRESHOLD = 0.05

# Number of cycles before moving from stop sign.
STOP_COUNTS = 3

# COSTANTI PER OBSTACLE AVOIDANCE
NUM_SAMPLING_PIXEL_EMERGENCY_BRAKE = 5


# COSTANTI PER LA GESTIONE DEL SEMAFORO
GREEN = 0
RED = 2
DISTANCE_TO_TRAFFIC_LIGHT = 1#12.8#12.8
DISTANCE_TO_TRAFFIC_LIGHT_LEAVING = 25
DISTANCE_TO_INTERSECTION = 20#60
DISTANCE_TO_OBSTACLE = 22
MIN_DISTANCE_TO_OBSTACLE = 4


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, waypoints_start_intersection, map, vehicle, tf_list):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE

        #aggiunti precedentemente
        self._red_light                     = False
        self._front_obstacle                = False
        self._front_pedestrian              = False
        self._far_light                     = False
        self._near_light                    = False
        
        #NO ADDED
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0

        self._index_traffic                 = None
        self._traffic_light_state           = None
        self._distance_to_traffic_light     = None
        self._approaching_semaphore         = False
        self._lead_vehicle                  = False
        self._decelerate_lead_vehicle       = False
        self._close_vehicle                 = False
        self._left_lane                     = False
        self._leaving_semaphore             = False
        self._waypoints_start_intersection  = waypoints_start_intersection
        
        self._map = map
        self._vehicle = vehicle
        self._tf_list = tf_list
        self._last_tf_idx_passed = None
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead
        

    # Handles state transitions and computes the goal state.

        # def transition_state(self, waypoints, traffic_light_list, ego_state, closed_loop_speed, lines, shift):
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.

        args:
            waypoints: current waypoints to track (global frame).
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [Waypoint(), Waypoint()]
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states:
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations)
                              before moving from stop sign.
        """

        ### In questo stato si seguono i waypoints centrali alla strada con la velocità pari a desired_speed.
        ### In caso venga rilevato un pedone su uno dei path o viene rilevato un semaforo rosso abbastanza vicino o un veicolo
        # davanti troppo vicino si passa nello stato  DECELERATE_TO_STOP
        ### In caso venga rilevato un veicolo davanti si passa nello stato FOLLOW_LEAD
        ### In caso venga rilevato un semaforo rosso a distanza abbastanza elevata passo nello stato APPROACHING_SEMAPHORE

        if self._state == FOLLOW_LANE:

            print("FOLLOW LANE STATE")

            self._approaching_semaphore = False
            self._leaving_semaphore = False

            self._index_traffic = None
            self._traffic_light_state = None
            self._distance_to_traffic_light = None

            goal_index, _, _ = self.update_waypoints(waypoints, ego_state)
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            print("Goal index: ", goal_index)

            infos = check_for_traffic_light_within_distance_ahead(self._tf_list, self._vehicle, self._map,
                                                                  DISTANCE_TO_INTERSECTION)
            print(infos)
            if not infos is None:
                is_within_distance, traffic_light_index, distance_to_traffic_light = infos

            if self._front_obstacle or self._front_pedestrian:
                self._state = DECELERATE_TO_STOP

            ## Se mi trovo a una certa distanza da un'intersezione allora vado nello stato di approach al semaforo
            # TODO: Conversion in Carla 9
            # Controllare se la distanza dal prossimo tl è minore di DISTANCE_TO_INTERSECTION

            elif infos is not None and is_within_distance and traffic_light_index != self._last_tf_idx_passed:
                self._index_traffic = traffic_light_index
                self._traffic_light_state = GREEN if self._tf_list[
                                                         traffic_light_index].get_state() == carla.TrafficLightState.Green else RED  # Stato del semaforo (GREEN or RED)
                self._distance_to_traffic_light = distance_to_traffic_light
                self._state = APPROACHING_SEMAPHORE

        elif self._state == LEAVING_SEMAPHORE:
            print("############################LEAVING SEMAPHORE STATE")
            self._approaching_semaphore = False
            self._leaving_semaphore = True

            goal_index, _, _ = self.update_waypoints(waypoints, ego_state)
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # aggiorno la distanza dal semaforo precedentemente trovato
            if self._index_traffic != None:
                next_traffic_light = self._tf_list[self._index_traffic]
                object_location = get_trafficlight_trigger_location(next_traffic_light)
                object_waypoint = self._map.get_waypoint(object_location)

                distance_to_traffic_light = compute_target_distance(object_waypoint.transform,
                                                                    self._vehicle.get_transform())
                # distance_to_traffic_light = self.get_distance_from_traffic(traffic_lights_waypoints, self._index_traffic, ego_state)
                self._distance_to_traffic_light = distance_to_traffic_light
                if(self._distance_to_traffic_light >=0):
                    self._traffic_light_state = GREEN if next_traffic_light.get_state() == carla.TrafficLightState.Green else RED
                else:
                    self._traffic_light_state = GREEN

                print('##########################################')
                print(distance_to_traffic_light)

            if self._traffic_light_state == RED or self._front_obstacle or self._front_pedestrian:
                print("TL State: ", self._traffic_light_state)
                self._state = DECELERATE_TO_STOP

            elif abs(distance_to_traffic_light) > DISTANCE_TO_TRAFFIC_LIGHT_LEAVING:
                self._state = FOLLOW_LANE


        ### In questo stato, si procede ad una velocità più bassa visto che il veicolo si sta approcciando ad un semaforo
        ### In caso venga rilevato il semaforo rosso abbastanza vicino o un pedone si passa nello stato DECELERATE_TO_STOP
        ### In caso il semaforo diventi nel frattempo verde si passa nello stato FOLLOW_LANE
        elif self._state == APPROACHING_SEMAPHORE:
            print("############################APPROACHING SEMAPHORE STATE")

            self._approaching_semaphore = True

            goal_index, closest_len, closest_index = self.update_waypoints(waypoints, ego_state)
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # aggiorno la distanza dal semaforo precedentemente trovato
            next_traffic_light = self._tf_list[self._index_traffic]
            object_location = get_trafficlight_trigger_location(next_traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            distance_to_traffic_light = compute_target_distance(object_waypoint.transform,
                                                                self._vehicle.get_transform())
            # distance_to_traffic_light = self.get_distance_from_traffic(traffic_lights_waypoints, self._index_traffic, ego_state)
            self._distance_to_traffic_light = distance_to_traffic_light
            # print('DISTANZA DAL TRAFFIC LIGHT: ',str(distance_to_traffic_light))
            self._traffic_light_state = GREEN if next_traffic_light.get_state() == carla.TrafficLightState.Green else RED

            if self._front_pedestrian or self._front_obstacle:
                self._state = DECELERATE_TO_STOP

            elif self._index_traffic != None:
                # verify if traffic light is close enough and red, then get waypoint
                if round(self._distance_to_traffic_light,
                         1) <= DISTANCE_TO_TRAFFIC_LIGHT and self._traffic_light_state != GREEN:
                    goal_index_traffic = self.get_goal_index_to_traffic_light(waypoints, closest_len, closest_index,
                                                                              distance_to_traffic_light)
                    self._goal_index = goal_index_traffic
                    self._goal_state = waypoints[self._goal_index]
                    self._state = DECELERATE_TO_STOP

                # GREEN state
                elif round(self._distance_to_traffic_light,
                           1) <= DISTANCE_TO_TRAFFIC_LIGHT and self._traffic_light_state == GREEN:
                    # self._tf_list.remove(self._tf_list[self._index_traffic])
                    self._last_tf_idx_passed = self._index_traffic
                    self._state = LEAVING_SEMAPHORE



        ### In questo stato controlla se la velocità del veicolo è minore di una determinata soglia. In caso positivo si passa nello stato STAY_STOPPED
        ### Inoltre in caso il semaforo diventi verde e non sia presente nessun pedone che ostacola si passa nello stato FOLLOW_LANE
        elif self._state == DECELERATE_TO_STOP:
            print("DECELERATE TO STOP")

            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED

            elif not self._front_pedestrian and not self._front_obstacle and self._approaching_semaphore:
                # aggiorno la distanza dal semaforo precedentemente trovato
                next_traffic_light = self._tf_list[self._index_traffic]
                object_location = get_trafficlight_trigger_location(next_traffic_light)
                object_waypoint = self._map.get_waypoint(object_location)

                distance_to_traffic_light = compute_target_distance(object_waypoint.transform,
                                                                    self._vehicle.get_transform())

                self._distance_to_traffic_light = distance_to_traffic_light
                # print('DISTANZA DAL TRAFFIC LIGHT: ', str(distance_to_traffic_light))
                self._traffic_light_state = GREEN if next_traffic_light.get_state() == carla.TrafficLightState.Green else RED

                if self._index_traffic != None and round(self._distance_to_traffic_light,
                                                         1) > DISTANCE_TO_TRAFFIC_LIGHT:
                    self._state = APPROACHING_SEMAPHORE

            elif not self._front_pedestrian and not self._front_obstacle and self._leaving_semaphore:
                self._state = LEAVING_SEMAPHORE

            elif not self._front_pedestrian and not self._front_obstacle:
                self._state = FOLLOW_LANE


        ### In questo stato il veicolo controlla l'assenza di semaforo rosso, pedoni e veicoli vicini.
        ### In caso positivo si passa nello stato FOLLOW_LANE.
        elif self._state == STAY_STOPPED:
            print("STAY STOPPED!!!")
            if self._approaching_semaphore and not self._front_pedestrian and not self._front_obstacle:

                # aggiorno la distanza dal semaforo precedentemente trovato
                next_traffic_light = self._tf_list[self._index_traffic]
                object_location = get_trafficlight_trigger_location(next_traffic_light)
                object_waypoint = self._map.get_waypoint(object_location)
                print("TL Location: ", object_waypoint )
                distance_to_traffic_light = compute_target_distance(object_waypoint.transform,
                                                                    self._vehicle.get_transform())

                self._distance_to_traffic_light = distance_to_traffic_light
                print('DISTANZA DAL TRAFFIC LIGHT: ', str(distance_to_traffic_light))
                self._traffic_light_state = GREEN if next_traffic_light.get_state() == carla.TrafficLightState.Green else RED

                if self._index_traffic != None and round(self._distance_to_traffic_light,
                                                         1) > DISTANCE_TO_TRAFFIC_LIGHT:
                    self._state = APPROACHING_SEMAPHORE

                elif self._traffic_light_state == GREEN:
                    # self._waypoints_start_intersection.remove(self._waypoints_start_intersection[self._closest_index_intersection])
                    self._state = LEAVING_SEMAPHORE

            elif not self._front_pedestrian and not self._front_obstacle and self._leaving_semaphore:
                self._state = LEAVING_SEMAPHORE

            elif not self._front_pedestrian and not self._front_obstacle:
                self._state = FOLLOW_LANE

        else:
            raise ValueError('Invalid state value.')
        # self._state = FOLLOW_LANE

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)
                



    def update_waypoints(self,waypoints, ego_state):
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        try:
            while waypoints[goal_index][2] <= 0.1: goal_index += 1
        except IndexError:
            return

        self._goal_index = goal_index
        self._goal_state = waypoints[goal_index]

        return goal_index, closest_len, closest_index



    # Funzione che seleziona il semaforo più vicino al veicolo fra la lista di tutti i semafori messa a disposizione 
    # da Carla, scartando quelli che non sono presenti nell'angolo di vista del veicolo e che hanno un orientamento
    # non adeguato
    def find_closest_traffic_sign(self, traffic_lights_list, traffic_light_orientation, ego_state):
        closest_len = float('Inf')
        index = -1

        for i in range(len(traffic_lights_list)):
            temp = (traffic_lights_list[i].x - ego_state[0])**2 + (traffic_lights_list[i].y - ego_state[1])**2
            
            traffic_light_delta_vector = [traffic_lights_list[i].x - ego_state[0], traffic_lights_list[i].y - ego_state[1]]
            traffic_light_distance = np.linalg.norm(traffic_light_delta_vector)

            traffic_light_delta_vector = np.divide(traffic_light_delta_vector, traffic_light_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]

            # esclusione dei semafori che formano un angolo maggiore di 45 con l'angolo di orientamento del 
            # veicolo
            if np.dot(traffic_light_delta_vector, ego_heading_vector) < (1 / math.sqrt(2)):
                continue          
            
            # esclusione dei semafori che non hanno un orientamento giusto (per evitare di selezioanre semafori 
            # nell'altro senso di marcia)
            yaw_traffic = "{}".format(traffic_light_orientation[i]).split(' ')[1]
            yaw_traffic = float(yaw_traffic) * pi / 180
            yaw_ego = ego_state[2]

            difference_yaw =  yaw_traffic -  yaw_ego

            if difference_yaw > pi:
                difference_yaw -= 2*pi
            elif difference_yaw < -pi:
                difference_yaw += 2*pi
            
            if not -0.2 < difference_yaw - pi/2 < 0.2:
                continue 

            if temp < closest_len:
                index = i
                closest_len = temp

        closest_len = np.sqrt(closest_len)

        return closest_len, index



    def get_distance_from_traffic(self, traffic_lights_list, index_traffic, ego_state):
    
        distance = (traffic_lights_list[index_traffic].x - ego_state[0])**2 + \
                    (traffic_lights_list[index_traffic].y - ego_state[1])**2   
        distance = np.sqrt(distance)

        return distance
        

    def get_distance_from_waypoints(self, waypoints, ego_state):
    
        closest_len = float('Inf')
        closest_index = 0

        for i in range(len(waypoints)):
            temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
            if temp < closest_len:
                closest_len = temp
                closest_index = i
        closest_len = np.sqrt(closest_len)

        return closest_len, closest_index

    def get_goal_index_to_traffic_light(self, waypoints, closest_len, closest_index, depth_value):
        
        arc_length = closest_len
        wp_index = closest_index

        # The earliest waypoint is over the traffic light
        if arc_length > depth_value:
            return wp_index

        if wp_index == len(waypoints) - 1:
            return wp_index

        max_wp_index = closest_index
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)

            if arc_length < depth_value:
                max_wp_index = wp_index + 1
            else: 
                return max_wp_index

            wp_index += 1

        return max_wp_index % len(waypoints)
    



# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]:
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False


def traffic_light_state_to_string(t_state):
    if t_state == RED:
        return 'RED'
    elif t_state == GREEN:
        return 'GREEN'
    else:
        return 'YELLOW'

    
def state_to_string(state):
    
    if state == FOLLOW_LANE:
        return 'FOLLOW_LANE'
    elif state == LEAVING_SEMAPHORE:
        return 'LEAVING_SEMAPHORE'
    elif state == APPROACHING_SEMAPHORE:
        return 'APPROACHING_SEMAPHORE'
    elif state == DECELERATE_TO_STOP:
        return 'DECELERATE_TO_STOP'
    elif state == STAY_STOPPED:
        return 'STAY_STOPPED'
    
    
    
### MY FUNCTION
def compute_target_distance(target_transform, current_transform):
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)
    current_pos_tf = [target_transform.location.x , target_transform.location.y, target_transform.location.z]
    current_vehicle_pos = [current_transform.location.x , current_transform.location.y, current_transform.location.z]
    current_vehicle_rot = [math.radians(current_transform.rotation.roll), math.radians(current_transform.rotation.pitch), math.radians(current_transform.rotation.yaw)]
    tf=transform_world_to_ego_frame(current_pos_tf, current_vehicle_pos, current_vehicle_rot)
    print(tf)
    if (tf[0] < 0):
        return -norm_target
    return norm_target

        
        
def check_for_traffic_light_within_distance_ahead(traffic_light_list, vehicle, map, distance):
    traffic_light_index=None
    is_within_distance = False
    actual_distance = None
    for i,traffic_light in enumerate(traffic_light_list):
        object_location = get_trafficlight_trigger_location(traffic_light)
        #object_waypoint=map.get_waypoint(traffic_light.get_transform().location, True)#Metodo alternativo senza usare trigger coso
        object_waypoint = map.get_waypoint(object_location)
        ego_vehicle_waypoint = map.get_waypoint(vehicle.get_location())
        

        if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
            continue

        ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
        wp_dir = object_waypoint.transform.get_forward_vector()
        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp < 0:
            continue

        is_within_distance,actual_distance =  is_within_distance_ahead(object_waypoint.transform,
                                    vehicle.get_transform(),
                                    distance)

        #QUESTO PERCHE IS_WITHIN_DISTANCE MI DICE SOLO SE IL SEMAFORO STA DAVANTI A ME
        #MA NOI LO ABBIAMO USATO COME SE INDICASSE SE STA ANCHE A DISTANZA MINORE DI DISTANCE.
        if (is_within_distance and actual_distance<=distance):
            is_within_distance=True
            traffic_light_index = i
            print('SELEZIONATO SEMAFORO CON INDICE:', str(traffic_light_index))
        else:
            is_within_distance=False

        
    return (is_within_distance, traffic_light_index, actual_distance)
    
        

def get_trafficlight_trigger_location(traffic_light):  # pylint: disable=no-self-use
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """
    def rotate_point(point, radians):
        """
        rotate a given point by a given angle
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)

# trasforma un punto in coordinate mondo in coordinate veicolo
# prende in input la posizione, i dati del veicolo, relativo a coordinate e angoli correnti e
# ne restituisce la posizione rispetto al veicolo
def transform_world_to_ego_frame(pos, ego, ego_rpy):
    loc = np.array(pos) - np.array(ego)
    r = transforms3d.euler.euler2mat(ego_rpy[0], ego_rpy[1], ego_rpy[2]).T
    loc_relative = np.dot(r, loc)
    return loc_relative