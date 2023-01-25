import sys, os
import pybullet as p
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from functions import get_fingers_dict, path_starting_from_code, DEBUG_LEVEL, LOG_PATH
from logs import _log_handling as log

class _Fingers(object):

    logger = None

    def __init__(self,model, model_id, arm_model, finger_name):
        self.model_id = model_id
        self.arm_model = arm_model
            
        self._name = finger_name
        self._id = int(self._name.strip("finger_"))
        self.finger_config = get_fingers_dict(self.arm_model)[self._name]

        # Joints and links of the finger
        self.lid = self.get_links_ids()
        self.jid = self.get_joints_ids()

        self.joints = model.get_joints(self.jid)
        self.update_joints_pose()

        self.links = model.get_links(self.lid)

        # If there is interpenetration between the fingers and the object 
        # (hands is not in a good orientation for that finger to close)
        # we lock it into open position
        self.locked = False

        # Distance from the object
        self.distance = 10000


    def __str__(self) -> str:
        representation = "{} with: \n".format(self._name)

        for joint in self.joints:
            representation += " {} at {} \n".format(joint.name,p.getJointState(self.model_id, joint.jid)[0])

        return representation


    def init_log(self):
         
        path_log = path_starting_from_code(0) + LOG_PATH()

        #Initialisation du mode debug ou release
        # Choix du mode DEBUG ou RELEASE
        if(DEBUG_LEVEL() == log.DEBUG_LEVEL.DEBUG_SOFT):
            #log.config["filename_debug"] = path_log + "debug"
            log.config["loggername"] = "arm"
            _Fingers.logger = log.factory.create('DEBUG',**log.config)
        else:
            #log.config["filename_release"] = path_log + "release"
            log.config["loggername"] = "arm"
            _Fingers.logger = log.factory.create('RELEASE',**log.config)

    def add_segments(self, segment):
        pass

    def add_joint(self,joint):
        pass

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def get_joints_ids(self):
        jointsId = []
        numJoints = p.getNumJoints(self.model_id)
        joints_name = self.finger_config["joints"]["names"].split(",")

        for j in range(numJoints):
            if p.getJointInfo(self.model_id, j)[1].decode('UTF-8') in joints_name:
                jointsId.append(j)

        if(len(jointsId) < len(joints_name)):
            _Fingers.logger.error("Some joints were not found")
            sys.exit()

        return jointsId

    def get_links_ids(self):
        linksId = []
        numJoints = p.getNumJoints(self.model_id)
        links_name = self.finger_config["links"].split(",")

        for j in range(numJoints):
            if p.getJointInfo(self.model_id, j)[12].decode('UTF-8') in links_name:
                linksId.append(j)

        if(len(linksId) < len(links_name)):
            _Fingers.logger.error("Some fingers were not found")
            sys.exit()

        return linksId


    def update_joints_pose(self):
        open_pose = self.finger_config["joints"]["open"].split(",")
        close_pose = self.finger_config["joints"]["close"].split(",")

        for i,joint in enumerate(self.joints):
            joint.update_open_pose(open_pose[i])
            joint.update_close_pose(close_pose[i])

    def close(self):
        """
        Close the finger
        """
        for joint in self.joints:
            joint.set_position(joint.close_pose,joint.limits["force"])


    def open(self):
        """
        Open the finger
        """
        for joint in self.joints:
            joint.set_position_no_force(joint.open_pose)

    def IsJointsOpened(self) -> bool:
        res = True

        for joint in self.joints:
            current_pos = p.getJointState(self.model_id,joint.jid)[0]

            if((current_pos - joint.open_pose) > 0.1):
                res = False
                break

        return res


    def adjust(self,joint_pose:dict):
        for joint in self.joints:
            
            if(joint.close_pose == 0 and joint.open_pose == 0):
                joint.set_position_no_force(joint.open_pose)
            else:
                if(joint.name in joint_pose.keys()):
                    pos = max(joint.limits["lower"], min(joint_pose[joint.name], joint.limits["upper"]))
                    # Modifier la force pour pas être en dur
                    joint.set_position_no_force(pos)#set_position_no_force(pos) or #set_position(pos,20)#joint.limits["force"])


    def axiscreator(self, linkId = -1):
        linkId = self.lid[0]
        # print(f'axis creator at bodyId = {self.robotId} and linkId = {linkId} as XYZ->RGB')
        x_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0] ,
                                    lineToXYZ            = [0.1, 0, 0],
                                    lineColorRGB         = [1, 0, 0] ,
                                    lineWidth            = 0.1 ,
                                    lifeTime             = 0 ,
                                    parentObjectUniqueId = self.model_id ,
                                    parentLinkIndex      = linkId )

        y_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0],
                                    lineToXYZ            = [0, 0.1, 0],
                                    lineColorRGB         = [0, 1, 0],
                                    lineWidth            = 0.1,
                                    lifeTime             = 0,
                                    parentObjectUniqueId = self.model_id,
                                    parentLinkIndex      = linkId)

        z_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                    lineToXYZ            = [0, 0, 0.1],
                                    lineColorRGB         = [0, 0, 1]  ,
                                    lineWidth            = 0.1        ,
                                    lifeTime             = 0          ,
                                    parentObjectUniqueId = self.model_id     ,
                                    parentLinkIndex      = linkId     )
        return [x_axis, y_axis, z_axis]