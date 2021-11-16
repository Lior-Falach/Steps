#!/usr/bin/env python3

from time import time
import numpy as np
import rospy
from rospy import Publisher, Rate
from std_msgs.msg import String
from unitree_legged_msgs.msg import A1LowState
from datetime import datetime
import os
import scipy.interpolate
import time
import logging

import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error

from locomotion.agents.whole_body_controller import com_velocity_estimator
from locomotion.agents.whole_body_controller import gait_generator as gait_generator_lib
from locomotion.agents.whole_body_controller import locomotion_controller
from locomotion.agents.whole_body_controller import openloop_gait_generator
from locomotion.agents.whole_body_controller import raibert_swing_leg_controller
from locomotion.agents.whole_body_controller import torque_stance_leg_controller

from locomotion.robots import a1
from locomotion.robots import a1_robot
from locomotion.robots import robot_config
from locomotion.robots.gamepad import gamepad_reader

# Params
_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.

_STANCE_DURATION_SECONDS = [
                               0.3
                           ] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


class RobotControl:

    def __init__(self, rate: int = 10, show_gui: bool = False, use_real_robot: bool = False,
                 logdir: str = '', max_time_secs: int = 30):

        # Init node
        rospy.init_node('robot_control', anonymous=True)

        # Node's attributes
        self.max_time_secs: int = max_time_secs
        self.use_real_robot: bool = use_real_robot
        self._rate: Rate = Rate(rate)  # hz
        self._low_state_publisher: Publisher = Publisher('/low_state', A1LowState, queue_size=10)  # todo: queue_size?

        # Construct simulator
        if show_gui and not self.use_real_robot:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.setPhysicsEngineParameter(numSolverIterations=30)
        self.p.setTimeStep(0.001)
        self.p.setGravity(0, 0, -9.8)
        self.p.setPhysicsEngineParameter(enableConeFriction=0)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.loadURDF("plane.urdf")

        # Construct robot class:
        if self.use_real_robot:
            self.robot = a1_robot.A1Robot(
                pybullet_client=self.p,
                motor_control_mode=robot_config.MotorControlMode.HYBRID,
                enable_action_interpolation=False,
                time_step=0.002,
                action_repeat=1)
        else:
            self.robot = a1.A1(self.p,
                               motor_control_mode=robot_config.MotorControlMode.HYBRID,
                               enable_action_interpolation=False,
                               reset_time=2,
                               time_step=0.002,
                               action_repeat=1)

        self.controller = self._setup_controller()
        self.controller.reset()

        # Init log directory
        self.logdir = logdir
        if self.logdir:
            self.logdir = os.path.join(self.logdir,
                                       datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            os.makedirs(self.logdir)

    def _setup_controller(self):
        """Demonstrates how to create a locomotion controller."""
        desired_speed = (0, 0)
        desired_twisting_speed = 0

        gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            self.robot,
            stance_duration=_STANCE_DURATION_SECONDS,
            duty_factor=_DUTY_FACTOR,
            initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
            initial_leg_state=_INIT_LEG_STATE)
        window_size = 20 if not self.use_real_robot else 60
        state_estimator = com_velocity_estimator.COMVelocityEstimator(
            self.robot, window_size=window_size)
        sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self.robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_height=self.robot.MPC_BODY_HEIGHT,
            foot_clearance=0.01)

        st_controller = torque_stance_leg_controller.TorqueStanceLegController(
            self.robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_body_height=self.robot.MPC_BODY_HEIGHT)

        controller = locomotion_controller.LocomotionController(
            robot=self.robot,
            gait_generator=gait_generator,
            state_estimator=state_estimator,
            swing_leg_controller=sw_controller,
            stance_leg_controller=st_controller,
            clock=self.robot.GetTimeSinceReset)

        return controller

    def _update_controller_params(self, lin_speed, ang_speed):

        self.controller.swing_leg_controller.desired_speed = lin_speed
        self.controller.swing_leg_controller.desired_twisting_speed = ang_speed
        self.controller.stance_leg_controller.desired_speed = lin_speed
        self.controller.stance_leg_controller.desired_twisting_speed = ang_speed

    def run(self):

        """ Run the node """

        start_time = self.robot.GetTimeSinceReset()
        current_time = start_time
        states, actions = [], []

        while current_time - start_time < self.max_time_secs:
        # while (not rospy.is_shutdown()) and (current_time - start_time < self.max_time_secs):

            start_time_robot = current_time
            start_time_wall = time.time()

            # Updates the controller behavior parameters.
            lin_speed, ang_speed, e_stop = self._generate_example_linear_angular_speed(current_time)
            # print(lin_speed)

            if e_stop:
                logging.info("E-stop kicked, exiting...")
                break

            self._update_controller_params(lin_speed, ang_speed)
            self.controller.update()
            hybrid_action, info = self.controller.get_action()
            states.append(
                dict(timestamp=self.robot.GetTimeSinceReset(),
                     base_rpy=self.robot.GetBaseRollPitchYaw(),
                     motor_angles=self.robot.GetMotorAngles(),
                     base_vel=self.robot.GetBaseVelocity(),
                     base_vels_body_frame=self.controller.state_estimator.
                     com_velocity_body_frame,
                     base_rpy_rate=self.robot.GetBaseRollPitchYawRate(),
                     motor_vels=self.robot.GetMotorVelocities(),
                     contacts=self.robot.GetFootContacts(),
                     qp_sol=info['qp_sol']))
            actions.append(hybrid_action)
            self.robot.Step(hybrid_action)
            current_time = self.robot.GetTimeSinceReset()

            if not self.use_real_robot:
                expected_duration = current_time - start_time_robot
                actual_duration = time.time() - start_time_wall
                if actual_duration < expected_duration:
                    time.sleep(expected_duration - actual_duration)

            # Publish the robot state
            # self._low_state_publisher.publish(*self._a1.low_state())

            # Sleep
            # self._rate.sleep()

    @staticmethod
    def _generate_example_linear_angular_speed(t):
        """Creates an example speed profile based on time for demo purpose."""
        vx = 0.4
        vy = 0.2
        wz = 0.8

        time_points = (0, 5, 10, 15, 20, 25, 30)
        speed_points = ((0, 0, 0, 0), (vx, 0, 0, 0), (-vx, 0, 0, 0), (0, 0, 0, -wz),
                        (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

        speed = scipy.interpolate.interp1d(time_points,
                                           speed_points,
                                           kind="previous",
                                           fill_value="extrapolate",
                                           axis=0)(t)

        return speed[0:3], speed[3], False


if __name__ == '__main__':

    a1_control = RobotControl(rate=100, use_real_robot=True, max_time_secs=10)

    try:
        a1_control.run()
    except rospy.ROSInterruptException:
        pass
