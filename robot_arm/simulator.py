import time
from pprint import pprint
import mujoco
from mujoco.rollout import rollout
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('/home/nmelgiri/workspace/src/robot_arm/robot_arm/robot.mjcf')
data = mujoco.MjData(model)

# pprint(dir(model))
# pprint(dir(model.body(20)))
# pprint(dir(model.joint("joint_1")))

# for body in model.body_names:
    # print(body)

state = None
for i in range(100):
  state, sensor_data = rollout(model, data, state if state else None)
  pprint(state)

