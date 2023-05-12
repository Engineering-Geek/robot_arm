import time

import mujoco
import mujoco.viewer


class Arm:
    def __init__(self, robot_xml_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(robot_xml_path)
        self.data = mujoco.MjData(self.model)
        
        


