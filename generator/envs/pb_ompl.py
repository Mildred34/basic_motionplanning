import sys, os
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

import pybullet as p
import utils
import time
from itertools import product
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from generator.ressources.arm import Arm

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are actuated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.

    - id
    - num_dim : number of joints non-fixed
    - joint_idx : joint non-fixed ids or that I don't want to plan
    - joint_bounds : joint min and max value
    - state : current joints value
    '''
    def __init__(self,arm : Arm) -> None:

        # Public attributes
        self.id = arm.robotId

        # prune fixed joints
        all_joint_num = p.getNumJoints(self.id)
        self.all_joint_idx = list(range(all_joint_num))

        self.end_effector_joint_id = arm.endeffectorId
        joint_idx = [j for j in self.all_joint_idx if (self._is_not_fixed(j) and j <= self.end_effector_joint_id)]

        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx

        print(self.joint_idx)
        self.joint_bounds = []

        self.reset_state()

    def _is_not_fixed(self, joint_idx):
        """
        Return True if joint not fixed ; False otherwise
        """
        joint_info = p.getJointInfo(self.id, joint_idx)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):

            joint_info = p.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds

            if low < high:
                self.joint_bounds.append([low, high])

        print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        """
        Return current joints value
        """
        return copy.deepcopy(self.state)

    def update_joints(self) -> None:

        joint_idx = [j for j in self.all_joint_idx if (self._is_not_fixed(j) and j <= self.end_effector_joint_id)]

        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx


    def set_joint_effector_id(self,jointid : int) -> None:
        """
            Set joint effector Id
            And update joints in the motion planning algorithm
        """
        self.end_effector_joint_id = jointid
        self.update_joints()

    def set_state(self, state):
        '''
        Set robot state.
        Doesn't Use dynamics ; Preferable to use it at start
        To faciliate collision checking

        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset_state(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        """
        Mets tous les joints à 0
        """
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        """
        - num_dim : number of joint actuated
        - state_sampler : How we're going to sample or State-space
        """

        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        # Allocate an instance of the default uniform state sampler for this space
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, robot, obstacles = []) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            robot_id : id given by pybullet for our robot
            obstacles: list of obstacle ids. Optional.
            space : State space + dimension of our State space + sampler...
            ss :  define and solve your motion planning problem
            si: The created space information
            check_link_pairs : pairs of self links to check for collision (self-collision checking)
            check_body_pairs : pairs of links - obstacle to check for collision

        '''
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()

        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])

        # Limit our state space knowing the joints values extremums
        self.space.setBounds(bounds)

        #  define and solve your motion planning problem
        self.ss = og.SimpleSetup(self.space)

        # Set the function, to check if the State is good (is the state inside the obstacle space ?)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

        # Get the current instance of the space information
        self.si = self.ss.getSpaceInformation()

        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("RRT") # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        """
        satisfy bounds TODO
        Should be unecessary if joint bounds is properly set
        """

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):

        # Self collision check list
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []

        # A set of all the moving links ; can not be modified afterwise
        moving_links = frozenset(
            [item for item in utils.get_moving_links(robot.id, robot.joint_idx) if not item in allow_collision_links])

        moving_bodies = [(robot.id, moving_links)]

        # Pair of links and obstacles to check if there is a collision
        # [[robot_id,[links],obstacle_links]]
        self.check_body_pairs = list(product(moving_bodies, obstacles)) #

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME):
        '''
            Plan a path to goal from the given robot start state

        '''
        print("start_planning")

        # if(debug()):
        #     print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)

        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []

        if solved:
            print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            sol_path_geometric.interpolate(INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()

            # list of each state to follow before till the goal state
            sol_path_list = [self.state_to_list(state) for state in sol_path_states] 
            # print(len(sol_path_list))
            # print(sol_path_list)

            # Check if the path found is valid
            for sol_path in sol_path_list:
                self.is_state_valid(sol_path)

            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time = DEFAULT_PLANNING_TIME):
        '''
        plan a path to goal from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time)

    def execute(self, path, dynamics=True):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i],force=5 * 240.)
            else:
                self.robot.set_state(q)

            p.stepSimulation()
            time.sleep(0.01)



    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        """
        If we want to change the default state sampler
        """
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        """
        Turn into list the state of each moving joint of the robot
        """
        return [state[i] for i in range(self.robot.num_dim)]