#!/usr/bin/env python3
import pybullet as p
import os
import pybullet_data # you can provide your own data or data package from pybullet

class Plane:
    def __init__(self, client : int):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self._id = p.loadURDF("plane.urdf")

def main():
    val = (
        __file__.replace(os.path.dirname(__file__), "")[1:]
        + " is meant to be imported not executed"
    )
    print(f"\033[91m {val}\033[00m")


if __name__ == "__main__":
    main()