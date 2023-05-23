from copy import copy
import time
from typing import Union

import numpy as np

import mujoco
import mujoco.viewer
import mujoco.renderer
import mediapy as media
from tqdm.auto import tqdm


class Arm:
    def __init__(self, robot_mjcf_path: str, kp: float, kd: float, ki: float) -> None:
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_error = np.zeros(3)
        self.model = mujoco.MjModel.from_xml_path(robot_mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.renderer.Renderer(self.model)

        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joints = [self.model.joint(i) for i in range(self.model.njnt - 2)]
        self.end_effector_joints = [self.model.joint(i) for i in range(self.model.njnt - 2, self.model.njnt)]

        mujoco.mj_kinematics(self.model, self.data)
        self.target = None
        for body_id in range(self.model.nbody):
            if "target" in self.model.body(body_id).name:
                self.target = self.model.body(body_id)
        assert self.target is not None, "End effector's target not found in the model."
        self._jacobian_position = np.zeros((3, self.model.njnt))
        self._jacobian_rotation = np.zeros((3, self.model.njnt))
        self.jacobian = np.zeros((6, self.model.njnt))
        self.previous_jacobian = np.zeros((6, self.model.njnt))
        self.dt = self.model.opt.timestep
        self.mass_matrix = np.zeros((self.model.nv, self.model.nv))
        self._gravity_matrix = np.zeros(self.model.nv)
        self.step_index = -1

        self.update_jacobian()
        self.update_mass_matrix()
        self.previous_location = self.data.site_xpos[0]

        self.errors = []

    def update_jacobian(self) -> None:
        """
        Updates the jacobian of the end effector (both position and rotation).
        Returns: None
        """
        mujoco.mj_jac(self.model, self.data, self._jacobian_position, self._jacobian_rotation, self.data.site_xpos[0],
                      self.target.id)
        jacobian_position = copy(self._jacobian_position).reshape(3, self.model.njnt)
        jacobian_rotation = copy(self._jacobian_rotation).reshape(3, self.model.njnt)
        self.previous_jacobian = copy(self.jacobian)
        self.jacobian = np.concatenate((jacobian_position, jacobian_rotation), axis=0)

    def update_mass_matrix(self) -> None:
        mujoco.mj_fullM(self.model, self.mass_matrix, self.data.qM)

    @property
    def coriolis_and_gravitational_forces(self):
        return self.data.qfrc_bias

    def simulate(self, real_time_factor: float = 1.0, max_time: float = 10.0):

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < max_time:
                step_start = time.time()
                self._step()
                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step / real_time_factor)

    def _step(self) -> None:
        """
        Steps the simulation forward by one timestep.
        Returns: None
        """
        if self.step_index > 0:
            self.data.ctrl[:] = self.controller()
        mujoco.mj_step(self.model, self.data)
        self.update_jacobian()
        self.update_mass_matrix()
        self.step_index += 1

    def _torque(self, target_position: np.array, target_velocity: np.array):
        """The torque needed will be calculated using the following equations:

        .. math::
            \dot{q}_{target} = W^{-1}J^{T}(JW^{-1}J^{T})^{-1}\dot{x}_{target}

        Args:
            target_position:
            target_velocity:

        Returns:

        """
        pos_error = target_position - self.data.site_xpos[0]
        vel_error = target_velocity - (self.data.site_xpos[0] - self.previous_location) / self.dt
        self.integral_error += pos_error * self.dt
        control_error = self.kp * pos_error + self.kd * vel_error + self.ki * self.integral_error
        mass_matrix = self.mass_matrix
        jacobian = self.jacobian[:3, :]
        inv_mass_matrix = np.linalg.inv(mass_matrix)
        q_dot_desired = inv_mass_matrix @ jacobian.T @ (jacobian @ inv_mass_matrix @ jacobian.T) @ control_error
        q_dot_dot_desired = (q_dot_desired - self.data.qvel) / self.dt
        applied_torque = mass_matrix @ (q_dot_dot_desired + self.kd * (q_dot_desired - self.data.qvel)) + \
                         self.coriolis_and_gravitational_forces
        return applied_torque

    def end_effector_location(self):
        return self.data.site_xpos[0]

    def controller(self) -> np.ndarray:
        """
        Returns the control input to the robot.
        Returns: np.ndarray
        """
        raise NotImplementedError
