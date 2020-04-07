## Implementation of solo bullet environment to test the 
## Deep Q stepper
## Author : Avadesh Meduri
## Date : 7/4/2020

import numpy as np

import time

import os
import rospkg
import pybullet as p
import pinocchio as se3
from pinocchio.utils import se3ToXYZQUAT

from robot_properties_solo.config import Solo12Config
from robot_properties_solo.quadruped12wrapper import Quadruped12Robot

from py_blmc_controllers.solo_impedance_controller import SoloImpedanceController 
from py_blmc_controllers.solo_centroidal_controller import SoloCentroidalController

from py_utils.trajectory_generator import TrajGenerator
from py_utils.solo_state_estimator import SoloStateEstimator


from pinocchio.utils import zero


class SoloBulletEnv:

    def __init__(self, ht, step_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com):
        '''
        Input:
            ht : height of the COM above the ground
            step_time : the duration after which a step is taken
            kp : p gain in leg impedance
            kd : d gain in leg impedance
            kp_com : p gain in wbc
            kd_com : d gain in wbc
            kp_ang_com : p gain orientation control
            kd_ang_com : d gain orientation control
        '''
        self.robot = Quadruped12Robot(ifrecord=False)
        self.total_mass = sum([i.mass for i in self.robot.pin_robot.model.inertias[1:]])

        self.ht = ht
        self.step_time = step_time
        self.dt = 0.001
        # Impedance controller initialisation
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,) 
        self.kp = 4 * kp
        self.kd = 4 * kd
        self.solo_leg_ctrl = SoloImpedanceController(self.robot)

        # Centroidal controller initialisation
        self.centr_controller = SoloCentroidalController(self.robot.pin_robot, self.total_mass,
        mu=0.5, kc=kp_com, dc=kd_com, kb=kp_ang_com, db=kd_ang_com,
        eff_ids=self.robot.pinocchio_endeff_ids)
        self.F = np.zeros(12)

        # Trajectory Generator initialisation
        self.trj = TrajGenerator(self.robot.pin_robot)
        self.f_lift = 0.1 ## height the foot lifts of the ground

        # State estimation initialisation
        self.sse = SoloStateEstimator(self.robot.pin_robot)
    
    def reset_env(self):
        q0 = np.matrix(Solo12Config.initial_configuration).T
        dq0 = np.matrix(Solo12Config.initial_velocity).T
        self.robot.reset_state(q0, dq0)

        self.xd_des = 4*[0,0,0]
        
        self.x_ori = [0., 0., 0., 1.]
        self.x_angvel = [0., 0., 0.]
        
        q, dq = self.robot.get_state()
        self.fl_foot, self.fr_foot, self.hl_foot, self.hr_foot = self.sse.return_foot_locations(q,dq)
        self.fl_off, self.fr_off, self.hl_off, self.hr_off = self.sse.return_hip_offset(q, dq)
        self.n = 0
        self.t = 0

        com = np.reshape(np.array(q[0:3]), (3,))
        dcom = np.reshape(np.array(dq[0:3]), (3,))
        u = np.average(self.fr_foot, self.hl_foot)

        return np.subtract(com[0:2], u[0:2]), dcom[0:2]

    def generate_traj(self, q, dq, fl_foot, fr_foot, hl_foot, hr_foot, n, u_t, t):
        '''
        This function returs the desired location of the foot for the 
        given time step
        '''

        x_des = 4*[0.0, 0.0, -self.ht]

        fl_hip, fr_hip, hl_hip, hr_hip = self.sse.return_hip_locations(q, dq)
        fl_off, fr_off, hl_off, hr_off = self.sse.return_hip_offset(q, dq)
        if np.power(-1, n) < 0: ## fl, hr leave the ground
            cnt_array = [0, 1, 1, 0]
            
            u_t_des = [[u_t[0], u_t[1], 0.0], [fr_foot[0], fr_foot[1], 0.0], [hl_foot[0], hl_foot[1], 0.0], [u_t[0], u_t[1], 0.0]]
            u_t_des[0] = np.add(u_t_des[0], fl_off) # assumption
            u_t_des[3] = np.add(u_t_des[3], hr_off) # assumption
            x_des[0:3] = np.subtract(self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],0.0], u_t_des[0], [0.0, 0.0, self.f_lift], self.step_time,t), [fl_hip[0], fl_hip[1], self.ht])
            x_des[3:6] = np.subtract(self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],0.0], u_t_des[1], [0.0, 0.0, 0.0], self.step_time,t), [fr_hip[0], fr_hip[1], self.ht])
            x_des[6:9] = np.subtract(self.trj.generate_foot_traj([hl_foot[0], hl_foot[1],0.0], u_t_des[2], [0.0, 0.0, 0.0], self.step_time,t), [hl_hip[0], hl_hip[1], self.ht])
            x_des[9:12] = np.subtract(self.trj.generate_foot_traj([hr_foot[0], hr_foot[1],0.0], u_t_des[3], [0.0, 0.0, self.f_lift], self.step_time,t), [hr_hip[0], hr_hip[1], self.ht])

        elif np.power(-1, n) > 0: ## fr and hl leave the ground
            cnt_array = [1, 0, 0, 1]
            
            u_t_des = [[fl_foot[0], fl_foot[1], 0.0], [u_t[0], u_t[1], 0.0], [u_t[0], u_t[1], 0.0], [hr_foot[0], hr_foot[1], 0.0]]
            u_t_des[1] = np.add(u_t_des[1], fr_off) # assumption
            u_t_des[2] = np.add(u_t_des[2], hl_off) # assumption
            x_des[0:3] = np.subtract(self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],0.0], u_t_des[0], [0.0, 0.0, 0.0], self.step_time,t), [fl_hip[0], fl_hip[1], self.ht])
            x_des[3:6] = np.subtract(self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],0.0], u_t_des[1], [0.0, 0.0, self.f_lift], self.step_time,t), [fr_hip[0], fr_hip[1], self.ht])
            x_des[6:9] = np.subtract(self.trj.generate_foot_traj([hl_foot[0], hl_foot[1],0.0], u_t_des[2], [0.0, 0.0, self.f_lift], self.step_time,t), [hl_hip[0], hl_hip[1], self.ht])
            x_des[9:12] = np.subtract(self.trj.generate_foot_traj([hr_foot[0], hr_foot[1],0.0], u_t_des[3], [0.0, 0.0, 0.0], self.step_time,t), [hr_hip[0], hr_hip[1], self.ht])

        return x_des, cnt_array
    
    def step_env(self, action, des_com, des_vel, x_ori, x_angvel):
        '''
        This function simulates the environment for step time duration
        Input:
            action : step length to be taken
        '''
        self.n += 1
        q, dq = self.robot.get_state()
        fl_foot, fr_foot, hl_foot, hr_foot = self.sse.return_foot_locations(q, dq) ## computing current location of the feet

        for t in range(int(self.step_time/self.dt)):
            p.stepSimulation()
            time.sleep(0.0001)

            q, dq = self.robot.get_state()
            com = np.reshape(np.array(q[0:3]), (3,))
            dcom = np.reshape(np.array(dq[0:3]), (3,))

            u_t = np.add(com[0:2], action)

            x_des, cnt_array = self.generate_traj(q, dq, fl_foot, fr_foot, hl_foot, hr_foot, self.n, u_t, self.dt*t)

            w_com = self.centr_controller.compute_com_wrench(self.t, q, dq, des_com, des_vel, x_ori, x_angvel)
            w_com[2] += self.total_mass * 9.81
            F = self.centr_controller.compute_force_qp(self.t, q, dq, cnt_array, w_com)

            tau = self.solo_leg_ctrl.return_joint_torques(q,dq,self.kp,self.kd,x_des, self.xd_des,F)
            self.robot.send_joint_command(tau)
    
            self.t += 1

        return np.subtract(com[0:2], u_t), dcom[0:2]