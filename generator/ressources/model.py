from typing import List
import pybullet as p
import numpy as np
import random
from functions import DEBUG_JOINTS, DEBUG_POS, GUI, path_starting_from_code, LOG_PATH, DEBUG_LEVEL
from logs import _log_handling as log

class Model(object):

    logger = None

    def __init__(self,path :str,start_pos : list,start_orn : list, scaling : float = 1, static : bool = True):
        self.load_model(path, start_pos, start_orn, scaling, static)

    def load_model(self, path, start_pos=[0, 0, 0], 
                   start_orn=[0, 0, 0, 1], scaling=1., static=False):

        if path.endswith('.sdf'):
            model_id = p.loadSDF(path, globalScaling=scaling)[0]
            p.resetBasePositionAndOrientation(model_id, start_pos, start_orn)
        else:
            model_id = p.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static)   
                     
        self.model_id = model_id
        # self._get_limits(self.model_id)


        self.joints, self.links = self.init_JointsandLinks()

        return model_id

    def init_log(self):
         
        path_log = path_starting_from_code(0) + LOG_PATH()

        #Initialisation du mode debug ou release
        # Choix du mode DEBUG ou RELEASE
        if(DEBUG_LEVEL() == log.DEBUG_LEVEL.DEBUG_SOFT):
            #log.config["filename_debug"] = path_log + "debug"
            log.config["loggername"] = "model"
            Model.logger = log.factory.create('DEBUG',**log.config)
        else:
            #log.config["filename_release"] = path_log + "release"
            log.config["loggername"] = "model"
            Model.logger = log.factory.create('RELEASE',**log.config)

    def init_JointsandLinks(self):

        joints, links = {}, {}

        for i in range(p.getNumJoints(self.model_id)):
            joint_info = p.getJointInfo(self.model_id, i)
            joint_name = joint_info[1].decode('utf8')
            joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
                            'force': joint_info[10],'damping': 0.1, 'speed': joint_info[11]}
            joints[i] = _Joint(self.model_id, i, joint_limits,joint_name)
            # link_name = joint_info[12].decode('utf8')
            links[i] = _Link(self.model_id, i)

        return joints, links


    def get_joints(self, jid : int) -> List["_Joint"]:
        sjoint = []

        if(self.joints):
            for i in range(p.getNumJoints(self.model_id)):
                if(self.joints[i].jid in jid):
                    sjoint.append(self.joints[i])
        else:
            print("Error when trying to get the joint")

        return sjoint

    def get_links(self, lid : int) -> List["_Link"]:
        slink = []

        if(self.links):
            for i in range(p.getNumJoints(self.model_id)):
                if(self.links[i].lid in lid):
                    slink.append(self.links[i])
        else:
            print("Error when trying to get the link")

        return slink

    def get_pose(self):
        """Return the pose of the model base."""
        pos, orn, _, _, _, _ = p.getLinkState(self.model_id, 3)
        return (pos, orn)
    
    def getBase(self):
        return p.getBasePositionAndOrientation(self.model_id)

class _Link(object):
    def __init__(self, model_id : int, link_id : int):
        self.model_id = model_id
        self.lid = link_id

    def get_pose(self):
        link_state = p.getLinkState(self.model_id, self.lid)
        position, orientation = link_state[0], link_state[1]
        return position, orientation


class _Joint(object):
    def __init__(self, model_id : int, joint_id : int, limits : dict, joint_name : str):
        self.model_id = model_id
        self.name = joint_name
        self.jid = joint_id
        self.limits = limits
        self.ranges = self.limits["upper"] - self.limits["lower"]

        self.rest_pose = random.uniform(self.limits["upper"],self.limits["lower"])

        self.open_pose = 0
        self.close_pose = 0


    def update_open_pose(self,open_pose):
        if open_pose == "zero":
            self.open_pose = 0
        else:
            self.open_pose = self.limits[open_pose]

    def update_close_pose(self,close_pose):
        if close_pose == "zero":
            self.close_pose = 0
        else:
            self.close_pose = self.limits[close_pose]

    def get_position(self) -> list:
        joint_state = p.getJointState(
            self.model_id, self.jid)
        return joint_state[0]

    def set_position(self, position, max_force : float = 20):

        self.disable_motor()

        max_force = np.clip(max_force, -self.limits["force"], self.limits["force"])

        p.setJointMotorControl2(
            self.model_id, self.jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            force=max_force,
            targetVelocity = 0.,
            positionGain   = 1) #1
            # maxVelocity=self.limits["speed"],
            # velocityGain=1)

    def set_position_no_force(self, position):

        self.disable_motor()

        p.setJointMotorControl2(
            self.model_id, self.jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            targetVelocity = 0.,
            positionGain   = 1) #1
            # maxVelocity=self.limits["speed"],
            # velocityGain=1)

    def disable_motor(self):
        p.setJointMotorControl2(
            self.model_id, self.jid, controlMode=p.VELOCITY_CONTROL, force=0.)