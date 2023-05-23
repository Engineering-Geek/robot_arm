from arm import Arm
import numpy as np
from sympy import Matrix


class NullPointArm(Arm):
    """
    This arm does its best to follow the equation given. The equation must produce a 3 point coordinate given a timestep
    value.
    """

    def __init__(self, point: np.array, *args, **kwargs):
        self.point = point
        super().__init__(*args, **kwargs)

    def controller(self) -> np.ndarray:
        """
        Find the null space of the Jacobian

        Returns:
            np.ndarray: The control input to the robot, shape: (n_joints, )

        """
        # rref = self.jacobian.rref()
        null_space = Matrix(self.jacobian.copy()).nullspace()[0]
        null_space = np.array(null_space).astype(np.float64)
        null_space = null_space / np.linalg.norm(null_space)
        null_space = null_space * 0.1
        torque = self._torque(self.point, null_space) / 2
        return torque


def main():
    arm = NullPointArm(
        point=np.array([0.0, 0.0, 1.0]),
        ki=100.0,
        kp=50.0,
        kd=50.0,
        robot_mjcf_path="/home/nmelgiri/workspace/src/robot_arm/robot_arm/robot.mjcf",
    )
    arm.simulate(real_time_factor=1.0, max_time=60.0)


if __name__ == "__main__":
    main()
