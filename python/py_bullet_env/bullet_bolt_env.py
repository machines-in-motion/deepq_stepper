## Implementation of botl bullet environment to test the 
## Deep Q stepper
## Author : Avadesh Meduri
## Date : 5/5/2020


import numpy as np
import time
import pybullet as p
import pinocchio as se3
from pinocchio.utils import se3ToXYZQUAT

from robot_properties_bolt.config import BoltConfig
from robot_properties_bolt.bolt_wrapper import BoltRobot

from py_blmc_controllers.bolt_impedance_controller import BoltImpedanceController
from py_blmc_controllers.bolt_centroidal_controller import BoltCentroidalController

from py_utils.trajectory_generator import TrajGenerator
from py_utils.bolt_state_estimator import BoltStateEstimator

from pinocchio.utils import zero

class BoltBulletEnv:

    def __init__(self, ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, ifrecord = False):
        '''
        Input:
            ht : height of the COM above the ground
            step_time : the duration after which a step is taken
            stance_time : duration of double support
            kp : p gain in leg impedance
            kd : d gain in leg impedance
            kp_com : p gain in wbc
            kd_com : d gain in wbc
            kp_ang_com : p gain orientation control
            kd_ang_com : d gain orientation control
        '''

        self.ht = ht
        self.step_time = step_time
        self.stance_time = stance_time
        self.dt = 0.001
        
        self.robot = BoltRobot()
        self.total_mass = sum([i.mass for i in self.robot.pin_robot.model.inertias[1:]])

        # Impedance controller iniitialization
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,) 
        self.kp = 2 * kp
        self.kd = 2 * kd

        self.bolt_leg_ctrl = BoltImpedanceController(self.robot)

        # Centroidal controller initialisation
        self.centr_controller = BoltCentroidalController(self.robot.pin_robot, self.total_mass,
        mu=0.2, kp=kp_com, kd=kd_com, kpa=kp_ang_com, kda=kd_ang_com,
        eff_ids=self.robot.pinocchio_endeff_ids)
        self.F = np.zeros(12)


        # Trajectory Generator initialisation
        self.trj = TrajGenerator(self.robot.pin_robot)
        self.f_lift = 0.05 ## height the foot lifts of the ground

        # State estimation initialisation
        self.sse = BoltStateEstimator(self.robot.pin_robot)

    def update_gains(self, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com):
        self.kp = 2 * kp
        self.kd = 2 * kd
        self.centr_controller = BoltCentroidalController(self.robot.pin_robot, self.total_mass,
        mu=0.5, kp=kp_com, kd=kd_com, kpa=kp_ang_com, kda=kd_ang_com,
        eff_ids=self.robot.pinocchio_endeff_ids)
        
    def reset_env(self):
        q0 = np.matrix(BoltConfig.initial_configuration).T
        dq0 = np.matrix(BoltConfig.initial_velocity).T
        self.robot.reset_state(q0, dq0)

        self.xd_des = 2*[0,0,0]
        
        self.x_ori = [0., 0., 0., 1.]
        self.x_angvel = [0., 0., 0.]
        
        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.sse.return_foot_locations(q,dq)
        self.n = 1 #right leg on the ground
        self.t = 0

        com = np.reshape(np.array(q[0:3]), (3,))
        dcom = np.reshape(np.array(dq[0:3]), (3,))

        u = fr_foot
       
        return np.around(com[0:2].T, 2), np.around(dcom[0:2].T, 2), np.around(u[0:2], 2), np.power(-1, self.n + 1)

    def generate_traj(self, q, dq, fl_foot, fr_foot, n, u_t, t, stance_time):
        '''
        This function returs the desired location of the foot for the 
        given time step
        '''

        x_des = 2*[0.0, 0.0, 0]
        xd_des = 2*[0.0, 0.0, 0]
        
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        if t < self.step_time - stance_time:
            if np.power(-1, n) < 0: ## fr leave the ground
                cnt_array = [1, 0]

                u_t_des = [[fl_foot[0], fl_foot[1], 0.0], [u_t[0], u_t[1], 0.0]]
                x_des[0:3] = np.subtract(u_t_des[0], [fl_hip[0], fl_hip[1], self.ht])
                x_des[3:6], xd_des[3:6] = self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],0.0], u_t_des[1], [0.0, 0.0, self.f_lift], self.step_time - stance_time,t)
                x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], self.ht])
                
            if np.power(-1, n) > 0: # fl leaves the ground
                cnt_array = [0, 1]

                u_t_des = [[u_t[0], u_t[1], 0.0], [fr_foot[0], fr_foot[1], 0.0]]
                x_des[0:3], xd_des[0:3] = self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],0.0], u_t_des[0], [0.0, 0.0, self.f_lift], self.step_time - stance_time,t)
                x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], self.ht])
                x_des[3:6] = np.subtract(u_t_des[1], [fr_hip[0], fr_hip[1], self.ht])
        else:
            cnt_array = [1, 1]

        return x_des, xd_des, cnt_array

    def step_env(self, action, des_com, des_vel, x_ori, x_angvel):
        '''
        This function simulates the environment for step time duration
        Input:
            action : step location (COP)
        '''
        self.n += 1
        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.sse.return_foot_locations(q, dq) ## computing current location of the feet

        u_t = action

        for t in range(int(self.step_time/self.dt)):
            p.stepSimulation()
            time.sleep(0.001)

            q, dq = self.robot.get_state()
            com = np.reshape(np.array(q[0:3]), (3,))
            dcom = np.reshape(np.array(dq[0:3]), (3,))
            x_des, xd_des, cnt_array = self.generate_traj(q, dq, fl_foot, fr_foot, self.n, u_t, self.dt*t, self.stance_time)

            w_com = self.centr_controller.compute_com_wrench(q, dq, des_com, des_vel, x_ori, x_angvel)
            w_com[2] += self.total_mass * 9.81
            F = self.centr_controller.compute_force_qp(q, dq, cnt_array, w_com)
            tau = self.bolt_leg_ctrl.return_joint_torques(q,dq,self.kp,self.kd,x_des, xd_des,F)
            self.robot.send_joint_command(tau)
    
            self.t += 1

        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.sse.return_foot_locations(q, dq) ## computing current location of the feet
        if np.power(-1, self.n) < 0:
            u = fr_foot
        else:
            u = fl_foot
        return np.around(com[0:2].T, 2), np.around(dcom[0:2].T, 2), np.around(u[0:2], 2), np.power(-1, self.n + 1)

    def start_recording(self, file_name):
        self.robot.start_recording(file_name)

    def stop_recording(self):
        self.robot.stop_recording()