#!/usr/bin/env python3
import os,sys
import time
from ast import Dict, List
import numpy as np
import math
import pybullet as p
import pybullet_data # you can provide your own data or data package from pybullet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from generator.ressources.arm import Arm
from generator.ressources.plane import Plane
from generator.ressources.table import Table
from generator.ressources.objects.random_object import Goal
# from generator.grasps.quality import Contacts_Generator # HERE
import matplotlib.pyplot as plt
from typing import List, Tuple
from functions import DEBUG_JOINTS,DEBUG_POS, GUI, GRAVITY, ARM_MODEL, DEBUG_LEVEL, LOG_PATH, path_starting_from_code, QualityMode, axe
from logs import _log_handling as log
import generator.envs.pb_ompl as pb_ompl

class GraspingGeneratorEnv():
    logger = None

    def __init__(self):
        """
        - Client : id of the simulator client
        - Robot : id of the robot that will be summoned into the sim
        - gui : If you wanna see the simulation, or just a rendering at the end
        - arm : robot arm
        - obstacles : list ids of every obstacle to avoid
        - object : object to grab
        - axe : gonna orientate the object along that axe
        - start and goal : joint poses for the start and goal positions
        - robot : description of our robot to then send to our motion planning algo
        - pb_ompl_interface : handle the motion planning
        - setup : Initialization being done ?
        """

        if( GraspingGeneratorEnv.logger is None):
            self.init_log()

        if(GUI()):
            GraspingGeneratorEnv.logger.debug("Graphique mode activated")
            self.gui = True
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
            p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0.4,0.4,0.7])
        else:
            self.gui = False
            self.client = p.connect(p.DIRECT)

        #if(DYNAMICS()):
        self.dynamics = True

        # engine Parameters
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        p.setPhysicsEngineParameter(solverResidualThreshold=0.001,numSolverIterations=300)
        p.setPhysicsEngineParameter(collisionFilterMode = 1, enableConeFriction = 1, contactSlop = 0.2)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1./240., self.client)
        # To see things clearly

        self.arm = None
        self.obstacles = []
        self.object = None

        # Quality test mode
        self.axe = axe.X
        self.reset()

        # Motion planning
        self.setup = False
        robot = pb_ompl.PbOMPLRobot(self.arm)
        self.robot = robot
        self.path = None
        self.index_path = 0

        # Setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

        # Debug parameters
        self.debug = {"joints":False,"pos": False}

        # Special modes
        if(self.gui):

            if(DEBUG_JOINTS()):
                GraspingGeneratorEnv.logger.debug("Joints mode activated")
                self.debug["joints"] = True
                self.angle = {}

                # Contrôle des articulations manuel en mode débug
                for joint_id in range(self.arm.numJoints):
                    if(p.getJointInfo(self.arm.robotId, joint_id)[2] != p.JOINT_FIXED):
                        joint_info = p.getJointInfo(self.arm.robotId, joint_id)
                        joint_type = joint_info[2]
                        joint_limits = {'lower': joint_info[8], 'upper': joint_info[9]}

                        if joint_type == p.JOINT_SPHERICAL:
                            self.angle[joint_id] =[p.addUserDebugParameter("%s pitch" % joint_info[1].decode('UTF-8'), joint_limits["lower"], joint_limits["upper"], 0),
                                                p.addUserDebugParameter("%s yaw" % joint_info[1].decode('UTF-8'), joint_limits["lower"], joint_limits["upper"], 0)] # No roll
                        else:
                            self.angle[joint_id] = p.addUserDebugParameter(p.getJointInfo(self.arm.robotId, joint_id)[1].decode('UTF-8'), joint_limits["lower"], joint_limits["upper"])

            elif(DEBUG_POS()):
                GraspingGeneratorEnv.logger.debug("Position mode activated")
                self.debug["pos"] = True
                self.action[0] = p.addUserDebugParameter('posX', -1, 1, 0.5)
                self.action[1] = p.addUserDebugParameter('posY', -1, 1, 0)
                self.action[2] = p.addUserDebugParameter('posZ', 0.5, 1.5, 1.0)


    def init_log(self):
         
        path_log = path_starting_from_code(0) + LOG_PATH()

        #Initialisation du mode debug ou release
        # Choix du mode DEBUG ou RELEASE
        if(DEBUG_LEVEL() == log.DEBUG_LEVEL.DEBUG_SOFT):
            #log.config["filename_debug"] = path_log + "debug"
            log.config["loggername"] = "env"
            GraspingGeneratorEnv.logger = log.factory.create('DEBUG',**log.config)
        else:
            #log.config["filename_release"] = path_log + "release"
            log.config["loggername"] = "env"
            GraspingGeneratorEnv.logger = log.factory.create('RELEASE',**log.config)

    def step(self, action : List = [0.6,0,0.8,0,0,0,1]):
        user_param = {}
        user_config = []
        res = None

        # Récupération de l'orientation/position du robot
        # if(debug):
        # robot_ob = self.arm.get_observation_quaternion(self.arm.endeffectorId)

        # Smooth simulation rendering to avoid jumping
        if(self.gui):
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        if(self.debug['joints']):
            for joint_id in range(self.arm.numJoints):

                joint_info = p.getJointInfo(self.arm.robotId, joint_id)
                joint_type = joint_info[2]

                if(joint_type != p.JOINT_FIXED):
                    if(joint_type != p.JOINT_SPHERICAL):
                        user_param["Joint_"+str(joint_id)] = p.readUserDebugParameter(self.angle[joint_id])
                    else:
                        angles_spherical = []

                        for id in self.angle[joint_id]:
                            angles_spherical.append(p.readUserDebugParameter(id))

                        user_param["Joint_"+str(joint_id)] = angles_spherical


            res = self.arm.apply_action(user_config,**user_param)

        elif(self.debug["pos"] and not self.debug['joints']):
            user_config.append(p.readUserDebugParameter(self.action[0]))
            user_config.append(p.readUserDebugParameter(self.action[1]))
            user_config.append(p.readUserDebugParameter(self.action[2]))
            res = self.arm.apply_action(user_config,**user_param)

            if(not res):
                GraspingGeneratorEnv.logger.warning("Inverse Kinematics not found")

        elif(not(self.debug["pos"] or self.debug["joints"])):

            # Set-up
            if(not self.setup):
                # Goal position to achieve
                # Domaine Cartésien
                # Convert Cartesian space to joint Space HERE
                #start = self.fromcartesiantojointspace([0,0,1,0,0,0,1])
                start = [0.,0.,0.,0.,0.,0.,0.]
                #goal = self.fromcartesiantojointspace([0,0.5,1.0,0,0,0,1])
                goal = [0.,0.463,0.344,-0.287,0.,0.243,0.514]
                self.robot.set_state(start)

                res, self.path = self.pb_ompl_interface.plan(goal) # Add planification in another thread
                self.setup = True

                if(not res):
                    GraspingGeneratorEnv.logger.error("Path not found!")
                    sys.exit()

            if(self.index_path < len(self.path)):
                if self.dynamics:
                    for i in range(self.robot.num_dim):
                        p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, self.path[self.index_path][i],force=5 * 240.)
                else:
                    self.robot.set_state(self.path[self.index_path])
                
                self.index_path = self.index_path + 1

        # show axis for end-effector and object
        if(self.gui):
            self.arm.axiscreator(linkId = -1)
            # The first obstacle is the object to grab
            self.object.axiscreator()

            for finger in self.arm._fingers:
                finger.axiscreator()

        p.stepSimulation()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # add object
        self.add_goal()

        # add other obstacles
        self.add_other_obstacles()

    def add_goal(self):
        # Set the goal to a random target
        # WIP Générer une position aléatoire dans le domaine atteignable du bras
        x = 0.5 # 0.4
        y = 0.35 # 0.4
        z = 0.5
             
        goal = (x, y, z)

        # Visual element of the goal
        if(self.object is None):
            self.object = Goal(self.client, goal, self.axe)
            GraspingGeneratorEnv.logger.info("Objet en cours d'étude:{}".format(self.object._name))
        else:
            self.object  = Goal(self.client, goal, self.axe)


        GraspingGeneratorEnv.logger.info("Axe en cours d'étude:{}".format(self.axe))


        self.obstacles.append(self.object.objId)

    def add_other_obstacles(self):
        # Reload the plane, and table
        p = Plane(self.client)
        t = Table(self.client)

        self.obstacles.append(p._id)
        self.obstacles.append(t._id)

    def add_robot(self):
        self.arm = Arm(self.client,ARM_MODEL())


    def reset(self):
        p.resetSimulation(self.client)

        if(GRAVITY()):
            p.setGravity(0, 0, -9.81)

        # Reload the plane, robot, table and object to grab
        self.clear_obstacles()
        self.add_obstacles()
        self.add_robot()
        
        

    def close(self):
        p.disconnect(self.client)

    def accurateCalculateInverseKinematics(self, targetPos : List, target_orient : List = None, threshold : float = 0.01, maxIter : int = 100) -> List:
        """
        Calcule la cinématique inversé pour atteindre la position ciblé nommé targetPos.
        Une loop avec un nombre d'itération max se fait afin d'atteindre au plus proche la cible.

        ---- WIP ----
            - Prise en compte des collisions
            - Eviter de s'approcher des positions singulières (quand on s'en rapproche, ca fait nimp par la suite)

        - targetPos : x,y,z
        - target_orient : x,y,z,w
        - threshold :
        - maxIter : 

        """
        closeEnough = False
        iter = 0
        distpos = 1e30
        distori = 1e30
        target_joint_positions = [0 for i in range(self.arm.numJoints)]

        while (not closeEnough and iter < maxIter):
            ik_joint_poses = p.calculateInverseKinematics(self.arm.robotId, self.arm.endeffectorId, targetPosition=targetPos, 
                targetOrientation=target_orient)

            j = 0
            k = 0 # Number of spherical joint


            for i in range(self.arm.numJoints): # Faire attention aux articulations:  "fixed base and fixed joints are skipped"
                joint_type = p.getJointInfo(self.arm.robotId, i)[2]

                if joint_type != p.JOINT_FIXED:
                    if joint_type != p.JOINT_SPHERICAL:
                        target_joint_positions[i] = ik_joint_poses[i-j+k*2]
                        p.resetJointState(self.arm.robotId, i, target_joint_positions[i])
                    else:
                        target_joint_positions[i] = [ik_joint_poses[i-j],ik_joint_poses[i-j+1], ik_joint_poses[i-j+2]]
                        p.resetJointStateMultiDof(self.arm.robotId, i, target_joint_positions[i])
                        k += 1
                else :
                    j += 1
                    target_joint_positions[i] = 0

            ls = p.getLinkState(self.arm.robotId, self.arm.endeffectorId)
            newPos = ls[4]
            newOri = ls[5]

            # On compare la distance
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            distpos = math.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))

            # Et l'orientation
            if type(target_orient)==list and len(target_orient) > 0:
                target_ang = p.getEulerFromQuaternion(target_orient)
                new_ang = p.getEulerFromQuaternion(newOri)
                diff = [target_ang[0] - new_ang[0], target_ang[1] - new_ang[1], target_ang[2] - new_ang[2]]
                distori = math.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
                
            closeEnough = (distpos < threshold) # and (distori < 1*math.pi/180)
            
            iter = iter + 1
        #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))

        if(closeEnough == False):
            GraspingGeneratorEnv.logger.debug("No Ik solution found!!!")
        
            
        return target_joint_positions, closeEnough

    def fromcartesiantojointspace(self,cartesianspace):
        """
        Turn Cartesian position into Joint Space with IK solver.

            - cartesianspace : cartesian positon to turn into joint space
        """

        target_joint_positions, closeEnough = self.accurateCalculateInverseKinematics(cartesianspace[0:3],cartesianspace[3:])
        target_valid = []

        # Si on a une erreur de cinématique, on reset la pos
        if(not closeEnough):
            GraspingGeneratorEnv.logger.error("Inverse Kinematics not found")
            sys.exit()
        else:
            for joint_id in self.arm.NonFixedJoints:
                if(joint_id not in self.arm._fingersJointsiD):
                    target_valid.append(target_joint_positions[joint_id])

        return target_valid 

def main():
    val = (
        __file__.replace(os.path.dirname(__file__), "")[1:]
        + " is meant to be imported not executed"
    )
    print(f"\033[91m {val}\033[00m")


if __name__ == "__main__":
    main()
