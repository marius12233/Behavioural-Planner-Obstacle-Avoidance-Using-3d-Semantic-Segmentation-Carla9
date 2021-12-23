import numpy as np
from math import cos, sin, pi,tan
import transforms3d

# Utils : X - Rotation
def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R

# Utils : Y - Rotation
def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R

# Utils : Z - Rotation
def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R

# Utils : Rotation - XYZ
def to_rot(r):
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx

# trasforma un punto in coordinate mondo in coordinate veicolo
# prende in input la posizione, i dati del veicolo, relativo a coordinate e angoli correnti e
# ne restituisce la posizione rispetto al veicolo
def transform_world_to_ego_frame(pos, ego, ego_rpy):
    loc = np.array(pos) - np.array(ego)
    r = transforms3d.euler.euler2mat(ego_rpy[0], ego_rpy[1], ego_rpy[2]).T
    loc_relative = np.dot(r, loc)
    return loc_relative


# classe che serve per proiettare punti da camera frame a vehicle frame senza dover ogni volta calcolarsi matrici intrinseca
# ed estrinseca
class ToWorldProjector:
    # inizializza matrice intrinseca ed estrinseca
    def __init__(self, camera_parameters):
        cam_height = camera_parameters['z']
        cam_x_pos = camera_parameters['x']
        cam_y_pos = camera_parameters['y']

        cam_yaw = camera_parameters['yaw']
        cam_pitch = camera_parameters['pitch']
        cam_roll = camera_parameters['roll']

        camera_width = camera_parameters['width']
        camera_height = camera_parameters['height']

        camera_fov = camera_parameters['fov']

        # Calcolo matrice intrinseca
        f = camera_width / (2 * tan(camera_fov * pi / 360))
        Center_X = camera_width / 2.0
        Center_Y = camera_height / 2.0

        intrinsic_matrix = np.array([[f, 0, Center_X],
                                     [0, f, Center_Y],
                                     [0, 0, 1]])

        self.inv_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

        #Matrice di rotazione per allineare image frame al camera frame
        rotation_image_camera_frame = np.dot(rotate_z(-90 * pi / 180), rotate_x(-90 * pi / 180))

        image_camera_frame = np.zeros((4, 4))
        image_camera_frame[:3, :3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]

        # Lambda Function per trasformare l'immagine da image frame a camera frame
        self.image_to_camera_frame = lambda object_camera_frame: np.dot(image_camera_frame, object_camera_frame)

        # Calcolo matrice estrinseca
        self.camera_to_vehicle_frame = np.zeros((4, 4))
        self.camera_to_vehicle_frame[:3, :3] = to_rot([cam_pitch, cam_yaw, cam_roll])
        self.camera_to_vehicle_frame[:, -1] = [cam_x_pos, cam_y_pos, cam_height, 1]

    # Effettua la proiezione dei punti dalla camera frame al vehicle frame ,
    # prendendo in input le coordinate e i dati di profondità e restituendo in output la loro proiezione
    def project(self,x,y,depth_data):
        pixel = [x, y, 1]
        pixel = np.reshape(pixel, (3, 1))

        # proiezione del pixel in image frame
        depth = depth_data[y][x] * 1000  # si considera la profondità in metri

        image_frame_vect = np.dot(self.inv_intrinsic_matrix, pixel) * depth

        # creazione di un vettore esteso
        image_frame_vect_extended = np.zeros((4, 1))
        image_frame_vect_extended[:3] = image_frame_vect
        image_frame_vect_extended[-1] = 1

        #proiezione da camera a vehicle frame
        camera_frame = self.image_to_camera_frame(image_frame_vect_extended)
        camera_frame = camera_frame[:3]
        camera_frame = np.asarray(np.reshape(camera_frame, (1, 3)))

        camera_frame_extended = np.zeros((4, 1))
        camera_frame_extended[:3] = camera_frame.T
        camera_frame_extended[-1] = 1

        vehicle_frame = np.dot(self.camera_to_vehicle_frame, camera_frame_extended)
        vehicle_frame = vehicle_frame[:3]
        vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1, 3)))

        return vehicle_frame
