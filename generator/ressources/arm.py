#!/usr/bin/env python3
# from ctypes import Union
import pybullet as p
import os, sys
import math
import pybullet_data # you can provide your own data or data package from pybullet
from typing import List, Dict, Tuple, Union
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from functions import DEBUG_JOINTS, DEBUG_POS, GUI, path_starting_from_code, LOG_PATH, DEBUG_LEVEL,ConvertEulerAngle2Quaternion, get_arm_dict, get_fingers_dict
from logs import _log_handling as log
from generator.ressources.model import Model
from generator.ressources.finger import _Fingers

class Arm:
    logger = None

    def __init__(self, client : int, arm_model : str):
        """
        
            - clientId :
            - start_pos :
            - start_orn :
            - arm_model :
            - robotId :
            - arm_config :
            - path :
            - endeffectorName :
            - _model :
            - _fingers :
            - _fingersJointsiD :
            - _joints :
            - numJoints :
            - endeffectorId : Id of end effector
            - NonFixedJoints : 
            - palmlinkname : 
            - _palmId : 
            - maxIter : 
            - ik_lower_limits :
            - ik_upper_limits : 
            - ik_joint_ranges : 
            - ik_rest_poses : 

        """

        if(Arm.logger == None):
            self.init_log()

        self.clientId = client

        # Position et orientation initiale du robot
        self.start_pos = [0,0,0]
        self.start_orn = p.getQuaternionFromEuler([0,0,0]) # Orientation must be in quaternion
        self.arm_model = arm_model
        
        # Add path
        self.robotId = None
        self.arm_config = get_arm_dict(self.arm_model)
        self.path = os.path.dirname(__file__) + self.arm_config["path"]
        self.endeffectorName = self.arm_config["endeffector"]

        # Chargement du modèle
        self._model = Model(self.path,self.start_pos,self.start_orn)
        self.robotId = self._model.model_id

        ## Configuration robot
        self._fingers = None
        self._fingersJointsiD = []
        self._joints = None
        self.numJoints = None
        self.endeffectorId = None
        self.NonFixedJoints = []
        self.palmlinkname = self.arm_config["palm"]["link"]
        self._palmId = None

        # Go to home pose
        #self.go_home()

        # Other parameters
        self.maxIter = 100 # Iteration on inverse kinematics calculation

        # Optional parameters for IK, read from urdf
        """ 
        (WIP)
        """
        #lower limits for null space
        self.ik_lower_limits = []
        #upper limits for null space
        self.ik_upper_limits = []
        #joint ranges for null space
        self.ik_joint_ranges = []
        #restposes for null space
        self.ik_rest_poses = []

        self.arm_configuration()

    def __str__(self) -> str:
        representation = "Arm {} with: \n".format(self.arm_model)

        for finger in self._fingers:
            representation += str(finger)

        return representation

    def init_log(self):
         
        path_log = path_starting_from_code(0) + LOG_PATH()

        #Initialisation du mode debug ou release
        # Choix du mode DEBUG ou RELEASE
        if(DEBUG_LEVEL() == log.DEBUG_LEVEL.DEBUG_SOFT):
            #log.config["filename_debug"] = path_log + "debug"
            log.config["loggername"] = "arm"
            Arm.logger = log.factory.create('DEBUG',**log.config)
        else:
            #log.config["filename_release"] = path_log + "release"
            log.config["loggername"] = "arm"
            Arm.logger = log.factory.create('RELEASE',**log.config)

    def get_ids(self) -> Tuple[int,int]:
        return self.robotId, self.clientId


    def get_handId(self) -> int:
        Id = 0

        for j in range(self.numJoints):
            Arm.logger.debug("Link name:{}".format(p.getJointInfo(self.robotId, j, physicsClientId=self.clientId)[12].decode('UTF-8')))

            # Retourne le nom du segment en byte
            if p.getJointInfo(self.robotId, j, physicsClientId=self.clientId)[12].decode('UTF-8') == self.endeffectorName:
                Id = j

        if(Id == 0):
            Arm.logger.error("End effector not found")
            sys.exit()

        return Id

    def get_palmId(self) -> int:
        Id = 0

        for j in range(self.numJoints):
            Arm.logger.debug("Link name:{}".format(p.getJointInfo(self.robotId, j, physicsClientId=self.clientId)[12].decode('UTF-8')))

            # Retourne le nom du segment en byte
            if p.getJointInfo(self.robotId, j, physicsClientId=self.clientId)[12].decode('UTF-8') == self.palmlinkname:
                Id = j

        if(Id == 0):
            Arm.logger.error("End effector not found")
            sys.exit()

        return Id

    def get_nonfixedjoints(self):
        for j in range(self.numJoints):
            if p.getJointInfo(self.robotId, j, physicsClientId=self.clientId)[2] != p.JOINT_FIXED:
                self.NonFixedJoints.append(j)

    def get_fingers(self):

        fingers = []
        i = 1
        fingers_config = get_fingers_dict(self.arm_model)


        for finger in fingers_config.keys() :
            fingers.append(_Fingers(self._model,self.robotId,self.arm_model,finger))
            i+=1

        try:
            assert(len(fingers) > 0)

        except AssertionError:
            Arm.logger.error("Fingers were not found")
            sys.exit(1) # 0 no error 1 error

        return fingers
    
    def is_there2many_fingerslocked(self):
        """
        Compte le nombre de doigt bloqué en position ouverte
        Si - de
        """
        # nombre de doigt en contact
        nb_fingers = len(self._fingers)
        nb_fingers_locked = 0
        res = False

        for finger in self._fingers:
            if(finger.locked):
                nb_fingers_locked += 1

        if((nb_fingers - nb_fingers_locked) < 2):
            res = True

        return res

    def unlockallfingers(self):
        for finger in self._fingers:

            if(finger.locked):
                Arm.logger.info("{} has been unlocked".format(finger._name))
                finger.locked = False

    def get_fingersJoints(self):
        jid = []

        for finger in self._fingers:
            jid.append(finger.jid)

        return jid

    def arm_configuration(self):
        self._joints = self._model.joints
        self.numJoints = p.getNumJoints(self.robotId)
        self.endeffectorId = self.get_handId()
        self._fingers = self.get_fingers()
        self._fingersJointsiD = self.get_fingersJoints()
        self._palmId = self.get_palmId()

        # Get non fixed joints
        self.get_nonfixedjoints()

        # Updates ik parameters
        self.update_ik_parameters(self.robotId, self.endeffectorId, False)

    def update_ik_parameters(self, body : int, target_joint : int, half_range : bool = False):
        
        # Configure IK optional parameters from joint info read from urdf file
        for jointId in range(self.numJoints):
            joint = self._joints[jointId]
            jointType = p.getJointInfo(body, jointId, physicsClientId=self.clientId)[2]

            if jointType != p.JOINT_FIXED:
                self.ik_lower_limits.append(joint.limits["lower"])
                self.ik_upper_limits.append(joint.limits["upper"])

                if not half_range:
                    self.ik_joint_ranges.append(joint.ranges)
                else:
                    self.ik_joint_ranges.append(joint.ranges/2)

                self.ik_rest_poses.append(joint.rest_pose)

                # self.ik_rest_poses[key].append((upper_limit + lower_limit)/2.0)


    def axiscreator(self, linkId = -1):
        linkId = self.endeffectorId
        # print(f'axis creator at bodyId = {self.robotId} and linkId = {linkId} as XYZ->RGB')
        x_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0] ,
                                    lineToXYZ            = [0.1, 0, 0],
                                    lineColorRGB         = [1, 0, 0] ,
                                    lineWidth            = 0.1 ,
                                    lifeTime             = 0 ,
                                    parentObjectUniqueId = self.robotId ,
                                    parentLinkIndex      = linkId )

        y_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0],
                                    lineToXYZ            = [0, 0.1, 0],
                                    lineColorRGB         = [0, 1, 0],
                                    lineWidth            = 0.1,
                                    lifeTime             = 0,
                                    parentObjectUniqueId = self.robotId,
                                    parentLinkIndex      = linkId)

        z_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                    lineToXYZ            = [0, 0, 0.1],
                                    lineColorRGB         = [0, 0, 1]  ,
                                    lineWidth            = 0.1        ,
                                    lifeTime             = 0          ,
                                    parentObjectUniqueId = self.robotId     ,
                                    parentLinkIndex      = linkId     )
        return [x_axis, y_axis, z_axis]

    def setJointAngleById(self, model, joint_id, joint_type, desired_angle ):

        if joint_type == p.JOINT_SPHERICAL:
            
            if(len(desired_angle) < 3):
                desired_angle.insert(0,0)

            desired_angle = ConvertEulerAngle2Quaternion(*desired_angle)
            p.setJointMotorControlMultiDof(bodyUniqueId=model,jointIndex=joint_id, controlMode=p.POSITION_CONTROL, targetPosition=desired_angle)

        elif joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
            p.setJointMotorControl2( bodyIndex=model, jointIndex=joint_id, controlMode=p.POSITION_CONTROL, targetPosition=desired_angle)


    def apply_action(self, position : List = None,**dic) -> bool:
        """
        Go to position ; step by step
        """

        res = False

        if(GUI() and DEBUG_JOINTS() and not DEBUG_POS()):

            try:
                assert(len(dic) > 0)

            except AssertionError:
                Arm.logger.error("No input given")
                sys.exit(-1)

            for joint_id in range(self.numJoints):

                joint_info = p.getJointInfo(self.robotId, joint_id)
                joint_type = joint_info[2]

                if(joint_type != p.JOINT_FIXED):

                    if("Joint_"+str(joint_id) not in dic.keys()):
                        Arm.logger.error("Joint not found in the position dictionnary!")
                        sys.exit()

                    self.setJointAngleById(self.robotId, joint_id, joint_type, dic["Joint_"+str(joint_id)])

            res = True
        else:
            if(position != None):


                target_joint_positions, close_enough = self.accurateCalculateInverseKinematics(targetPos = position[:3],target_orient = position[3:],
                      threshold = 0.01, maxIter = self.maxIter)

                if(close_enough):
                    res = True
                    for joint_id in self.NonFixedJoints:
                        if(joint_id not in self._fingersJointsiD):
                            joint_info = p.getJointInfo(self.robotId, joint_id)
                            joint_type = joint_info[2]

                            self.setJointAngleById(self.robotId, joint_id, joint_type, target_joint_positions[joint_id])

                            # p.setJointMotorControl2(
                            #                 bodyIndex      = self.robotId,
                            #                 jointIndex     = i,
                            #                 controlMode    = p.POSITION_CONTROL,
                            #                 targetPosition = target_joint_positions[i],
                            #                 targetVelocity = 0,
                            #                 force          = 1, # 500
                            #                 positionGain   = 1, # 1
                            #                 velocityGain   = 0.1)
                else:
                    #      val = "IK didn't find a solution close enough"
                    #      print(f"\033[91m {val}\033[00m")
                    pass


                if( not DEBUG_POS()):
                    # Close the grip # plus utilliser
                    if((dic["grip"])) and (dic["joint"] is None):
                        self.close_grip()

                    elif((dic["joint"] is not None) and 
                            (dic["grip"] != 0)):
                        # self.close_grip()
                        self.adjust_grip(dic["joint"])

                    # Open the grip
                    else:
                        self.open_grip()



        return res


    def get_observation_euler(self,link_index) -> tuple:
        """
        Get the position and orientation of any link in the arm in the simulations
        (COM frame not urdf frame)
        """
        result = p.getLinkState(bodyUniqueId = self.robotId, 
                                  linkIndex = link_index,
                                  physicsClientId = self.clientId)

        if(result is None):
            Arm.logger.error("None type found after observation!!!")

        pos = result[0]
        ang = result[1]

        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2])) # rotation z --> vecteur directeur projeté sur le plan (x,y)

        return pos+ang

    def get_observation_quaternion(self,link_index) -> tuple:
        """
        Get the position and orientation of any link in the arm in the simulations
        (COM frame not urdf frame)
        """
        result = p.getLinkState(bodyUniqueId = self.robotId, 
                                  linkIndex = link_index,
                                  physicsClientId = self.clientId)

        if(result is None):
            Arm.logger.error("None type found after observation!!!")

        pos = result[0]
        ang = result[1]

        return pos+ang

    def check_openfingers(self):
        res = True

        for finger in self._fingers:
            res = finger.IsJointsOpened()
            if res == False:
                break

        return res


    def open_grip(self):
        for finger in self._fingers:
            finger.open()

    def adjust_grip(self, joint_pos : dict):
        for finger in self._fingers:
            if not finger.locked : 
                finger.adjust(joint_pos[finger._name])
            else:
                finger.open()
            

    def close_grip(self):
        for finger in self._fingers:
            if not finger.locked : 
                finger.close()
            else:
                finger.open()

            # finger.set_position(finger.limits["upper"])


def main():
    val = (
        __file__.replace(os.path.dirname(__file__), "")[1:]
        + " is meant to be imported not executed"
    )
    print(f"\033[91m {val}\033[00m")


if __name__ == "__main__":
    main()









