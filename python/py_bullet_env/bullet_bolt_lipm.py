## Botl stepping environment with enforced lipm trajectory
## Tracks only with impedance
## Date : 28/05/2020
## Author : Avadesh Meduri

import numpy as np
import time
import pybullet as p
import pinocchio as se3
from pinocchio.utils import se3ToXYZQUAT

from robot_properties_bolt.config import BoltConfig
from robot_properties_bolt.bolt_wrapper import BoltRobot

from py_blmc_controllers.bolt_impedance_controller import BoltImpedanceController

from py_utils.trajectory_generator import TrajGenerator
from py_utils.bolt_state_estimator import BoltStateEstimator

from pinocchio.utils import zero


class BoltBulletLipm:

    def __init__(self, ht, step_time, kp, kd):
        '''
        Input:
            ht : height of the COM above the ground
            step_time : the duration after which a step is taken
            stance_time : duration of double support
            kp : p gain in leg impedance
            kd : d gain in leg impedance
        '''

        self.ht = ht
        self.step_time = step_time
        self.dt = 0.001

        self.robot = BoltRobot()
        self.total_mass = sum([i.mass for i in self.robot.pin_robot.model.inertias[1:]])

        # Impedance controller iniitialization
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,) 
        self.kp = 2 * kp
        self.kd = 2 * kd

        self.bolt_leg_ctrl = BoltImpedanceController(self.robot)

        self.f = 6* [0]
        self.f[2] = 0.5*(self.total_mass*9.81)
        self.f[5] = 0.5*(self.total_mass*9.81)

        # State estimation initialisation
        self.sse = BoltStateEstimator(self.robot.pin_robot)

        # Trajectory Generator initialisation
        self.trj = TrajGenerator(self.robot.pin_robot)
        self.f_lift = 0.05 ## height the foot lifts of the ground

        # A and B matrices to integrate LIPM dynamics
        self.omega = np.sqrt(9.81/self.ht)
        self.A = np.matrix([[1, 0, self.dt, 0], 
                            [0, 1, 0, self.dt], 
                            [self.dt*(self.omega**2), 0, 1, 0], 
                            [0, self.dt*(self.omega**2), 0, 1]])
            
        self.B = np.matrix([[0, 0], [0, 0], [-(self.omega**2)*self.dt, 0], [0, -(self.omega**2)*self.dt]])
            
    def update_gains(self, kp, kd):
        self.kp = 2 * kp
        self.kd = 2 * kd
        
    def reset_env(self):
        q0 = np.matrix(BoltConfig.initial_configuration).T
        dq0 = np.matrix(BoltConfig.initial_velocity).T
        self.robot.reset_state(q0, dq0)

        self.xd_des = 2*[0,0,0]
        
        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.sse.return_foot_locations(q,dq)
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        self.n = 1 #right leg on the ground
        self.t = 0

        com = np.reshape(np.array(q[0:3]), (3,))
        dcom = np.reshape(np.array(dq[0:3]), (3,))
        self.b = abs(fl_hip[1] - fr_hip[1])
        self.u = fr_foot
        
        return np.around(com[0:2].T, 2), np.around(dcom[0:2].T, 2), np.around(self.u[0:2], 2), np.power(-1, self.n + 1)

    def integrate_lipm(self, des_com, u):
        '''
        Computes the desired location and velocity of the COM assuming
        lipm dynamics.
        Input:
            des_com : current desired desired center of mass state
                        [x, xd, y, yd]
            u : current COP location
        '''
        des_com = np.matmul(self.A, des_com.T) + np.matmul(self.B, u.T)

        return np.ravel(des_com)

    def generate_traj(self, q, dq, fl_foot, fr_foot, des_com, n, u_t, t):
        '''
        This function generates the end effector trajectory
        '''
        x_des = 2*[0.0, 0.0, 0]
        xd_des = 2*[0.0, 0.0, 0]
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        des_hip_fl = [des_com[0], des_com[2] + 0.5*self.b, self.ht]
        des_hip_fr = [des_com[0], des_com[2] - 0.5*self.b, self.ht]
        
        if np.power(-1, n) < 0: ## fr leave the ground
            u_t_des = [[fl_foot[0], fl_foot[1], 0.0], [u_t[0], u_t[1], 0.0]]
            x_des[0:3] = np.subtract(u_t_des[0], des_hip_fl)
            xd_des[0:3] = [des_com[1], des_com[3], 0]
            x_des[3:6], xd_des[3:6] = self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],0.0], u_t_des[1], [0.0, 0.0, self.f_lift], self.step_time,t)
            x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], self.ht])

        if np.power(-1, n) > 0: ## fl leave the ground
            u_t_des = [[u_t[0], u_t[1], 0.0], [fr_foot[0], fr_foot[1], 0.0]]
            x_des[0:3], xd_des[0:3] = self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],0.0], u_t_des[0], [0.0, 0.0, self.f_lift], self.step_time,t)
            x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], self.ht])
            x_des[3:6] = np.subtract(u_t_des[1], des_hip_fr)
            xd_des[3:6] = [des_com[1], des_com[3], 0]

        return x_des, xd_des

    def step_env(self, action):
        '''
        This function simulates the environment for step time duration
        Input:
            action : step location (COP)
        '''

        self.n += 1
        q, dq = self.robot.get_state()
        com = np.reshape(np.array(q[0:3]), (3,))
        dcom = np.reshape(np.array(dq[0:3]), (3,))
        self.des_com = np.array([com[0], dcom[0], com[1], dcom[1]])

        fl_foot, fr_foot = self.sse.return_foot_locations(q, dq) ## computing current location of the feet
        u_t = action

        for t in range(int(self.step_time/self.dt)):
            p.stepSimulation()
            time.sleep(0.001)

            q, dq = self.robot.get_state()
            self.des_com = self.integrate_lipm(self.des_com, self.u[0:2])
            x_des, xd_des = self.generate_traj(q, dq, fl_foot, fr_foot, self.des_com, self.n, u_t, self.dt*t)    

            tau = self.bolt_leg_ctrl.return_joint_torques(q, dq, self.kp, self.kd, x_des, xd_des, self.f)
            self.robot.send_joint_command(tau)

            self.t += 1

        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.sse.return_foot_locations(q, dq) ## computing current location of the feet
        if np.power(-1, self.n) < 0:
            u = fr_foot
        else:
            u = fl_foot
        return np.around(com[0:2].T, 2), np.around(dcom[0:2].T, 2), np.around(u[0:2], 2), np.power(-1, self.n + 1)
