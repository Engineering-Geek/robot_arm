from arm import Arm
import numpy as np
from matplotlib import pyplot as plt


class TracerArm(Arm):
    """
    This arm does its best to follow the equation given. The equation must produce a 3 point coordinate given a timestep
    value.
    """

    def __init__(self, equation: callable, *args, **kwargs):
        self.equation = equation
        super().__init__(*args, **kwargs)

    def controller(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The control input to the robot, shape: (n_joints, )

        """
        time_input = self.dt * self.step_index
        target_position = self.equation(time_input)
        target_velocity = (self.equation(time_input + self.dt) - target_position) / self.dt
        torque = self._torque(target_position, target_velocity) / 2
        self.errors.append(np.linalg.norm(target_position - self.end_effector_location()))
        return torque


def eqn(t: float):
    """
    Draws a parabola in 3D space
    Args:
        t: The time value

    Returns:

    """
    # turn t into a value swinging between -1 and 1
    t = (t % 1) * 2 - 1
    x = 0
    y = t
    z = ((t - 1) * (t + 1)) + 1.5

    return np.array([x, y, z])


def main():
    arm = TracerArm(
        ki=100.0,
        kp=50.0,
        kd=50.0,
        equation=eqn,
        robot_mjcf_path="/home/nmelgiri/workspace/src/robot_arm/robot_arm/robot.mjcf",
    )
    arm.simulate(real_time_factor=1.0, max_time=30.0)
    # plot errors
    plt.plot(arm.errors)
    plt.show()


if __name__ == "__main__":
    main()

