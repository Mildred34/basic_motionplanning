from ast import List
from typing import Tuple
import pybullet as p
import os, sys
import pybullet_data # you can provide your own data or data package from pybullet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from functions import UseCase,axe, get_object_dict,Liste
import math
import generator.envs.pb_ompl as pb_ompl


class Goal:
    def __init__(self, client : int, base : Tuple, Axe : axe = axe.X, name : str = None):

        # Chargement de l'objet
        self._list = Liste()

        if(name is None):
            self._name = UseCase()
        else:
            self._name = name

        self._urdfname = self._name


        self.urdf_path = os.path.dirname(__file__) + "/" + self._list + "/" + self._urdfname + ".urdf"

        # Orientation de l'objet selon l'axe correspondant dans le sens donné
        if(Axe.value > Axe.Z):
            self.sens_orientation = 0
            self.Axe = axe(Axe.value-3)
        else:
            self.sens_orientation = 1
            self.Axe = Axe

        # Position and orientation properties
        self.worldframepos = []
        self.worldframeori = [] # En angle d'euler

        self.start_orn = self.orientation_alongAxe(self.Axe.name,self.sens_orientation)
        self.start_pos = [base[0], base[1], base[2]]

        # Get properties from .yaml
        # Longueurs caractéristiques selon chacun des axes ; W est la longueur peu importe la direction
        self.l = {'X':0,'Y':0,'Z':0, '-X':0, '-Y':0,'Z':0,'W':0} 
        # self.update_characteric_lengths()
        updated_startpos = self.update_position(self.start_pos[:],self.Axe)


        # Ids
        self.objId = p.loadURDF( self.urdf_path, basePosition = updated_startpos, baseOrientation = self.start_orn ,physicsClientId=client,
                 useMaximalCoordinates = True,useFixedBase = False)  
        self.clientId = client

        # init position and orientation
        self.get_observation()

        # Update Position 
        self.update_position(self.start_pos[:],self.Axe)

        # Change Dynamics of the object
        # p.changeDynamics(self.objId, -1, mass = 0)
        p.changeDynamics(self.objId, -1, lateralFriction = 0.9)
        p.changeDynamics(self.objId, -1, rollingFriction = 0.9)


    def update_position(self,origin,axe):
        """
        Place le bout de l'objet (par rapport à l'axe donné) au niveau de l'origine
        Vu que j'oriente l'objet dans l'axe X du repère monde. On ne modifie
        que la position en X
        """

        if(axe.name == "X"):
            if(self.sens_orientation):
                X = origin[0] + self.l["-X"]
            else:
                X = origin[0] + self.l["X"]

            origin[0] = X
        elif(axe.name == "Y"):
            if(self.sens_orientation):
                Y = origin[0] + self.l["-Y"]
            else:
                Y = origin[0] + self.l["Y"]
            origin[0] = Y
        else:
            if(self.sens_orientation): 
                Z = origin[0] + self.l["-Z"]
            else:
                Z = origin[0] + self.l["Z"]

            origin[0] = Z 

        return origin


    def reset_position(self):
        origin = self.start_pos
        if(self.Axe.name == "X"):
            X = origin[0] + self.l["X"]
            origin[0] = X
        elif(self.Axe.name == "Y"):
            Y = origin[0] + self.l["Y"]
            origin[0] = Y
        else: 
            Z = origin[0] + self.l["Z"] 
            origin[0] = Z 

        p.resetBasePositionAndOrientation(self.objId,origin,self.start_orn)

    def get_observation(self) -> List:
        # Get the position and orientation of the object in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.objId, self.clientId) # position et orientation du com
        ang2 = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2])) # rotation z --> vecteur directeur projeté sur le plan (x,y)

        self.worldframepos = list(pos)
        self.worldframeori = list(ang)

        return self.worldframepos+list(ang) # quaternion

    def axiscreator(self):
        linkId = 0
        # print(f'axis creator at bodyId = {self.objId} and linkId = {linkId} as XYZ->RGB')
        x_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0] ,
                                    lineToXYZ            = [0.1, 0, 0],
                                    lineColorRGB         = [1, 0, 0] ,
                                    lineWidth            = 0.1 ,
                                    lifeTime             = 0 ,
                                    parentObjectUniqueId = self.objId ,
                                    parentLinkIndex      = linkId )

        y_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0],
                                    lineToXYZ            = [0, 0.1, 0],
                                    lineColorRGB         = [0, 1, 0],
                                    lineWidth            = 0.1,
                                    lifeTime             = 0,
                                    parentObjectUniqueId = self.objId,
                                    parentLinkIndex      = linkId)

        z_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                    lineToXYZ            = [0, 0, 0.1],
                                    lineColorRGB         = [0, 0, 1]  ,
                                    lineWidth            = 0.1        ,
                                    lifeTime             = 0          ,
                                    parentObjectUniqueId = self.objId     ,
                                    parentLinkIndex      = linkId     )
        return [x_axis, y_axis, z_axis]

    def orientation_alongAxe(self, Axe : str = 'X', sens : bool = 1) -> list:
        quaternion = [0,0,0,1]

        if(Axe == "X"):

            if(sens):
                quaternion = self.get_quaternion([0,0,0])
            else:
                quaternion = self.get_quaternion([0,0,math.pi])

        elif(Axe == "Y"):

            if(sens):
                quaternion = self.get_quaternion([0,0,-math.pi/2])
            else:
                quaternion = self.get_quaternion([0,0,math.pi/2])

        elif(Axe == "Z"):

            if(sens):
                quaternion = self.get_quaternion([0,math.pi/2,0])
            else:
                quaternion = self.get_quaternion([0,-math.pi/2,0])

        else:
            print("Error: While finding an orientation")
            sys.exit()

        return quaternion

    def get_quaternion(self, euler_angle : list) -> list:
        orn = p.getQuaternionFromEuler(euler_angle)
        return orn


def main():
    val = (
        __file__.replace(os.path.dirname(__file__), "")[1:]
        + " is meant to be imported not executed"
    )
    print(f"\033[91m {val}\033[00m")


if __name__ == "__main__":
    main()
