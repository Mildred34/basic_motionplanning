#!/usr/bin/env python3

import yaml
import os, sys
import itertools
from typing import Dict, List, Union, Tuple
import ast
import math
from enum import IntEnum
import numpy as np

class axe(IntEnum):
    X = 0
    Y = X+1
    Z = Y+1
    Xbis = Z+1
    Ybis = Xbis+1
    Zbis = Ybis+1

def red_txt(txt: str) -> str:
    return f"\033[91m{txt}\033[00m"


def path_starting_from_code(go_back_n_times: int = 0) -> str:
    """
    Return the path of the parent directory of where the code is.
    Can also going backward to return an upper directory

    Input:
        - int : how much you want to go backward

    Output:
        - str : path of the wanted directory
    
    """
    path = os.path.dirname(__file__) # parent directory of where the program resides
    for _ in itertools.repeat(None, go_back_n_times):
        path = os.path.dirname(path) # going back in time, parent direction of the parent..;
    return path


def path_join_str(path: str, name: str) -> str:
    return os.path.join(path, name)


def __open_file_on_config_dir(file_name: str, extention: str = "yaml") -> Dict:
    """
    Charge un fichier yaml

    Input:
        - file_name (str)
        - extention (str) : default: yaml
------------
    Output: 
        - Dictionnaire avec les composantes du fichier yaml
    """
    ext = "."
    ext += extention
    with open(
        path_join_str(
            path_starting_from_code(),
            "config_files/" + file_name + ext,
        )
    ) as f:
        if extention == "yaml":
            return yaml.load(f, Loader=yaml.FullLoader)
        else:
            return ast.literal_eval(f.read())

def __return_config_dict() -> Dict:
    return __open_file_on_config_dir("config")


def __get_value_of_key_in_config_file(*key) -> str:
    config = __return_config_dict()

    for id in key:
        config = config[id]

    return config

def object_file_name(read: bool = True) -> str:
    if read:
        return __get_value_of_key_in_config_file("objet","path","read")
    else:
        return __get_value_of_key_in_config_file("objet","path","write")

def get_object_dict() -> Dict:
    """
    Obtention du dictionnaire du .yaml décrit dans le fichier config (config.yaml)
------------------------------------------------------------------
    Input: None
----------------
    Output:
        - Dictionnaire contenant le fichier Yaml 
    """
    objects = __open_file_on_config_dir(object_file_name()) 
    return objects

def get_objects_dict_withfilename(filename : str) -> Dict:
    """
        Obtention du dictionnaire du .yaml décrit dans le fichier filename
    ------------------------------------------------------------------
        Input: None
    ----------------
        Output:
            - Dictionnaire contenant le fichier Yaml 
        """
    res = True
    objects = None

    try:
        objects = __open_file_on_config_dir(filename) 

    except OSError:
        res = False
        
    return res, objects


def get_allarms_dict() -> Dict:
    """
    Obtention du dictionnaire du .yaml décrivant les configurations
    de tous les bras"
------------------------------------------------------------------
    Input: None
----------------
    Output:
        - Dictionnaire contenant le fichier Yaml 
    """
    arms = __open_file_on_config_dir("arm_config")
    return arms

def get_arm_dict(arm_model : str) -> Dict:
    """
    Obtention du dictionnaire du .yaml décrivant les configurations
    du bras de modèle "arm_model"
------------------------------------------------------------------
    Input: None
----------------
    Output:
        - Dictionnaire contenant le fichier Yaml 
    """
    arms = __open_file_on_config_dir("arm_config")[arm_model]
    return arms


def get_fingers_dict(arm_model : str) -> Dict:
    """
    Obtention du dictionnaire du .yaml décrivant les configurations
    des doigts du bras de modèle "arm_model"
------------------------------------------------------------------
    Input: None
----------------
    Output:
        - Dictionnaire contenant le fichier Yaml 
    """
    fingers = __open_file_on_config_dir("arm_config")[arm_model]["fingers"]
    return fingers


def __get_grasp_dict() -> Dict:
    objects = get_object_dict()
    result = {}
    for obj in objects:
        result[obj] = {"grasps": objects[obj]["grasps"]}
    return result


def __get_forces_dict() -> Dict:
    objects = get_object_dict()
    result = {}
    # On extrait les perturbations de chaque objet dans le .yaml
    # Norme des valeurs par défaut ?
    for obj in objects:
        result[obj] = {"perturbations": objects[obj]["perturbations"]}
    return result


def __get_stl_path_dict() -> dict:
    """
    Retourne les objets du fichier Yaml décrit par le fichier config.
    En ne retenant que :
     - stl path
     - characteristic length
     - center of mass

    Input: None

    Output: 
        - Dictionnaire
    """
    # Dictionnaire du yaml en config avec tous les objets
    objects = get_object_dict()

    # Path du dossier stl
    path = path_join_str(path_starting_from_code(1), "stl/")

    for obj in objects:
        # On retire grasp et perturbation du dico de chaque objet
        objects[obj].pop("grasps")
        objects[obj].pop("perturbations")

        # On rajoute le nom de l'objet au path du dossier stl
        objects[obj]["stl path"] = path_join_str(
            path, str(objects[obj]["stl file name"]) + ".stl"
        )

        # On supprime du dictionnaire la clé "stl file name"
        objects[obj].pop("stl file name")

    return objects

def ARM_MODEL() -> str:
    return str(__get_value_of_key_in_config_file("robot","ArmModel"))

def GUI() -> bool:
    return bool(__get_value_of_key_in_config_file("simulation","GUI"))

def DEBUG_JOINTS() -> bool:
    return bool(__get_value_of_key_in_config_file("debug","JOINTS"))

def DEBUG_POS() -> bool:
    return bool(__get_value_of_key_in_config_file("debug","POS"))

def GRAVITY() -> bool:
    return bool(__get_value_of_key_in_config_file("simulation","GRAVITY"))

def UseCase() -> str:
    return str(__get_value_of_key_in_config_file("objet","UseCase"))

def Liste() -> str:
    return str(__get_value_of_key_in_config_file("objet","Liste"))

def DEBUG_LEVEL() -> int:
    return int(__get_value_of_key_in_config_file("debug","LEVEL"))

def LOG_PATH() -> str:
    return str(__get_value_of_key_in_config_file("debug","LOG_PATH"))

def QualityMode() -> int:
    return int(__get_value_of_key_in_config_file("Quality","mode"))

def AutoMode() -> bool:
    return bool(__get_value_of_key_in_config_file("Quality","auto"))


### Saving Funtions ###
def green_txt(txt: str) -> str:
    return f"\033[92m{txt}\033[00m"
    
def save_yaml(name_of_file: str, dictionary: Dict):
    with open(
        path_join_str(
            path_starting_from_code(), "config_files/" + name_of_file + ".yaml"
        ),
        "w",
    ) as f:
        f.write(yaml.dump(dictionary, sort_keys=False))
    print(green_txt(name_of_file + ".yaml saved"))


### Mathematics functions ###

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)

    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = list(n / mag for n in v)

    return v

def normalize_quaternion(q : list, tolerance : float = 0.00001) -> list:
    try:
        assert(len(q) == 4)
    except AssertionError:
        sys.exit("Error: vector given was not a quaternion")

    mag2 = sum(n * n for n in q)

    if abs(mag2 - 1.0) > tolerance:
        q = list(n / mag2 for n in q)

    return q


def ConvertAxisAngle2Quaternion(axisangle:list):
    """
    Axis-angle must be defined as follow : 
    (theta,x,y,z)
    """

    try:
        assert(len(axisangle) == 4)
    
    except AssertionError:
        print("Not enough parameters")
        exit()


    theta = axisangle[0]
    v = normalize(axisangle[1:])

    w = math.cos(theta/2)
    q1 = v[0]*math.sin(theta/2)
    q2 = v[1]*math.sin(theta/2) 
    q3 = v[2]*math.sin(theta/2)

    return [q1,q2,q3,w]


def ConvertEulerAngle2Quaternion(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format

    According to what I see, it's a yaw, pitch, roll convention here
    It was written in the pybullet docs that they were doing roll,pitch then yaw
    But anyway...
    """

    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
 
    return [qx, qy, qz, qw]

def quaternion_multiply(quaternion1,quaternion0):
    """
    qs = q1*q0
    --> Quaterion défini par [q1,q2,q3,w]
    q1 = [r1,r2,r3,w1]
    q0 = [s1,s2,s3,w0]
    """
    w0 = quaternion0[3]
    s1 = quaternion0[0]
    s2 = quaternion0[1]
    s3 = quaternion0[2]
    w1 = quaternion1[3]
    r1 = quaternion1[0]
    r2 = quaternion1[1]
    r3 = quaternion1[2]

    return [w1*s1+r1*w0-r2*s3+r3*s2, # q1
            w1*s2+r1*s3+r2*w0-r3*s1, # q2
            w1*s3-r1*s2+r2*s1+r3*w0, # q3
            w1*w0-r1*s1-r2*s2-r3*s3] # w0


def quaternion_conjugate(q) -> list:
    x, y, z, w = q
    return [-x, -y, -z, w]

def quaternion_inversesign(q : list) -> list:
    x, y, z, w = q

    return [-x, -y, -z, -w]


def quaternion_2_rotationmatrix(q) -> np.ndarray:
    q1, q2, q3, q0 = q
    return np.array([[1-2*pow(q2,2)-2*pow(q3,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
                    [2*q1*q2+2*q0*q3, 1-2*pow(q1,2)-2*pow(q3,2), 2*q2*q3-2*q0*q1],
                    [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*pow(q1,2)-2*pow(q2,2)]])

def convertquaternion2rotationmatrixelements(q : list) -> tuple :
    """
    Given the rotation quaternion q = (q1,q2,q3,w), 
    we determine the corresponding rotation matrix elements
    """
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q0  = q[3]

    r11 = 1-2*pow(q2,2)-2*pow(q3,2)
    r12 = 2*q1*q2-2*q0*q3
    r13 = 2*q1*q3+2*q0*q2

    r21 = 2*q1*q2+2*q0*q3
    r22 = 1-2*pow(q1,2)-2*pow(q3,2)
    r23 = 2*q2*q3-2*q0*q1

    r31 = 2*q1*q3-2*q0*q2
    r32 = 2*q2*q3+2*q0*q1
    r33 = 1-2*pow(q1,2)-2*pow(q2,2)

    return r11, r12, r13, r21, r22, r23, r31, r32, r33

def qv_mult(q1,v1) -> list:

    v1.append(0.0) # on transforme le vecteur en quaternion
    q2 = v1

    # On retourne le vecteur transformé
    return quaternion_multiply(quaternion_multiply(q1,q2),quaternion_conjugate(q1))[:3] 


def AligntoAxe(prev_orientation:list, new_axe:str = 'X')-> list: 
    """
    prev_orientation is in quaternion !!

    Change the orientation to be aligned with the corresponding axis
    Orientation must be in quaternion.
    q = (q1,q2,q3,w)

    A rotation quaternion is similar to the axis-angle representation. If we know the axis-angle components (x,y,z,theta)
    we can convert to a rotation quaternion q as follow:
    w = cos(theta/2)
    q1 = x.sin(theta/2)
    q2 = y.sin(theta/2) 
    q3 = z.sin(theta/2)

    My arm is by default oriented along the z axis.
    """

    new_orientation = prev_orientation


    if new_axe == 'X': # rotation 90° along y axis - on oriente l'axe z sur le x
        rot90y = ConvertAxisAngle2Quaternion([math.pi/2,0,1,0])
        new_orientation = quaternion_multiply(rot90y,prev_orientation)

    elif new_axe =='Y': # rotation -90° along x axis - on oriente l'axe z sur le y
        # puis 90° sur Z pour aligner l'outil dans le bon sens (Franka)
        rot90x = ConvertAxisAngle2Quaternion([-math.pi/2,1,0,0])
        new_orientation = quaternion_multiply(rot90x,prev_orientation)
        rot90z = ConvertAxisAngle2Quaternion([math.pi/2,0,0,1])
        new_orientation = quaternion_multiply(rot90z,new_orientation)

    else:
        pass

    return new_orientation

def AligntoAxeEuler(prev_orientation:list, new_axe:axe = 'X')-> list: 

    new_orientation = prev_orientation
    sens = 1

    if(new_axe >= 3):
        sens = -1

    new_axe = axe(new_axe % 3)

    if new_axe.name == 'X': # rotation 90° along y axis - on oriente l'axe z sur le x
        if(sens == 1):
            pitch = 90*math.pi/180
            rot90y = ConvertEulerAngle2Quaternion(0, pitch, 0)
            new_orientation = quaternion_multiply(rot90y,prev_orientation)
        else:
            pitch = -90*math.pi/180
            rot90y = ConvertEulerAngle2Quaternion(0, pitch, 0)
            new_orientation = quaternion_multiply(rot90y,prev_orientation)

        
    elif new_axe.name =='Y': # rotation -90° along x axis - on oriente l'axe z sur le y
        # puis 90° sur Z pour aligner l'outil dans le bon sens (Franka)
        if(sens == 1):
            roll = -90*math.pi/180
            pitch = 0
            yaw = 0

            rot90x = ConvertEulerAngle2Quaternion(roll, pitch, yaw)
            new_orientation = quaternion_multiply(rot90x,prev_orientation)

            roll = 0
            pitch = 0
            yaw = 90*math.pi/180

            rot90z = ConvertEulerAngle2Quaternion(roll, pitch, yaw)
            new_orientation = quaternion_multiply(rot90z,new_orientation)
        else:
            roll = 90*math.pi/180
            pitch = 0
            yaw = 0

            rot90x = ConvertEulerAngle2Quaternion(roll, pitch, yaw)
            new_orientation = quaternion_multiply(rot90x,prev_orientation)

            roll = 0
            pitch = 0
            yaw = -90*math.pi/180

            rot90z = ConvertEulerAngle2Quaternion(roll, pitch, yaw)
            new_orientation = quaternion_multiply(rot90z,new_orientation)

    else:
        if(sens == 1):
            pass

        else:
            roll = 180*math.pi/180
            pitch = 0
            yaw = 0

            rot90x = ConvertEulerAngle2Quaternion(roll, pitch, yaw)
            new_orientation = quaternion_multiply(rot90x,prev_orientation)

            roll = 0
            pitch = 0
            yaw = -180*math.pi/180

            rot90z = ConvertEulerAngle2Quaternion(roll, pitch, yaw)
            new_orientation = quaternion_multiply(rot90z,new_orientation)

    return new_orientation

def average_quaternion(Q):
    """
    by Tolga Birdal
    Q is an Mx4 matrix of quaternions. weights is an Mx1 vector, a weight for
     each quaternion.
    Qavg is the weighted average quaternion
    This function is especially useful for example when clustering poses
    after a matching process. In such cases a form of weighting per rotation
    is available (e.g. number of votes), which can guide the trust towards a
    specific pose. weights might then be interpreted as the vector of votes 
    per pose.
    Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. 
    "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, 
    no. 4 (2007): 1193-1197.

    Slightly modified ; avoiding averaging quaternion
    might not be unique if there is several maximum eigen value
    in my program M would allways have unit coefficient
    I would not use it

    At the end we normalize the quaternion can be skipped if we want to speed up
    
    """

    pass

### Utilities 

def get_KeyIndex(key:str,undict: dict) -> int:

    index = -1

    if key in undict:
        index = list(undict).index(key)

    return index

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    return -1 


def main():
    val = (
        __file__.replace(os.path.dirname(__file__), "")[1:]
        + " is meant to be imported not executed"
    )
    print(f"\033[91m {val}\033[00m")


if __name__ == "__main__":
    main()