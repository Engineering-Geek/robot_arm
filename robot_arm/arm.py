from copy import copy
import time

import numpy as np

import mujoco
import mujoco.viewer


class Arm:
    def __init__(self, robot_mjcf_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(robot_mjcf_path)
        self.data = mujoco.MjData(self.model)

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
        self.step_index = 0

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

    def simulate(self,
                 steps: int = 100,
                 render: bool = True,
                 _time: bool = False,
                 real_time_factor: float = 1.0,
                 max_time: float = 10.0,
                 ) -> None:
        """
        Simulates the robot for the given number of steps.
        Args:
            steps: Number of steps to simulate.
            render: Whether to render the simulation.
            _time: Whether to print the time taken for each step.
            real_time_factor: The factor by which to slow down the simulation. 1.0 means real time, and 0.0 means as fast as possible.
            max_time: The maximum time to simulate for. If the simulation takes longer than this, it will be stopped.

        Returns:

        """
        start_time = time.time()
        times = []
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        for i in range(steps):
            self._step()
            if render:
                viewer.sync()
                time_taken = time.time() - start_time
                time_until_next_step = self.model.opt.timestep - time_taken % self.model.opt.timestep
                times.append(time_taken)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step / real_time_factor)
            if max_time is not None and time.time() - start_time > max_time:
                break
        if _time:
            print(f"Time taken: {sum(times)}")
            print(f"Average time per step: {sum(times) / len(times)}")

    def _step(self) -> None:
        """
        Steps the simulation forward by one timestep.
        Returns: None
        """
        self.data.ctrl[:] = self.controller()
        mujoco.mj_step(self.model, self.data)
        self.update_jacobian()
        self.update_mass_matrix()

    def controller(self) -> np.ndarray:
        """
        Returns the control input to the robot.
        Returns: np.ndarray
        """
        return np.ones(self.model.nu)


def main():
    arm = Arm("/home/nmelgiri/workspace/src/robot_arm/robot_arm/robot.mjcf")
    arm.simulate(100000, max_time=30.0)


if __name__ == "__main__":
    main()
