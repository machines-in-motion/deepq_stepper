## this is an implementation of the bullet bolt env with 
## spring loaded centroidal dynanmics motion planner
## Author : Avadesh Meduri
## Date : 28/06/2020

import numpy as np
from math import atan2, pi, sqrt
import time
import pybullet as p
import pinocchio as se3

from robot_properties_bolt.config import BoltConfig
from robot_properties_bolt.bolt_wrapper import BoltRobot

from py_blmc_controllers.bolt_impedance_controller import BoltImpedanceController
from py_blmc_controllers.bolt_centroidal_controller import BoltCentroidalController

from py_utils.trajectory_generator import TrajGenerator
from py_utils.bolt_state_estimator import BoltStateEstimator

from py_motion_planner.sl_cent_motion_planner import SLCentMotionPlanner

from pinocchio.utils import zero

from matplotlib import pyplot as plt


class BulletSLCBoltEnv:

    def __init__(self, ht, step_time, air_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, ifrecord = False):
        '''
        Input:
            ht : height of the COM above the ground
            step_time : the duration after which a step is taken
            air_time : duration of air phase
            kp : p gain in leg impedance
            kd : d gain in leg impedance
            kp_com : p gain in wbc
            kd_com : d gain in wbc
            kp_ang_com : p gain orientation control
            kd_ang_com : d gain orientation control
        '''

        self.ht = ht
        self.step_time = step_time
        self.air_time = air_time
        self.dt = 0.001
        
        self.robot = BoltRobot()
        self.total_mass = sum([i.mass for i in self.robot.pin_robot.model.inertias[1:]])
        q0 = np.matrix(BoltConfig.initial_configuration).T
        self.inertia = np.diag(self.robot.pin_robot.mass(q0)[3:6, 3:6])
        
        # Robot parameters to account for to match IPM training env
        # size of the foot diameter
        self.foot_size = 0.02
        # This is the distance of the hip from the true COM
        # This is subtracted as in the training the COM is assumed to be
        # at the hip joint with no offset 
        self.base_offset = 0.078 #check urdf for value
        # Impedance controller iniitialization
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,) 
        self.kp = 2 * kp
        self.kd = 2 * kd

        self.bolt_leg_ctrl = BoltImpedanceController(self.robot)

        # Centroidal controller initialisation
        self.centr_controller = BoltCentroidalController(self.robot.pin_robot, self.total_mass,
        mu=0.1, kp=kp_com, kd=kd_com, kpa=kp_ang_com, kda=kd_ang_com,
        eff_ids=self.robot.pinocchio_endeff_ids)
        self.F = np.zeros(6)

        # Trajectory Generator initialisation
        self.trj = TrajGenerator(self.robot.pin_robot)
        self.f_lift = 0.07 ## height the foot lifts of the ground
        
        # State estimation initialisation
        self.sse = BoltStateEstimator(self.robot.pin_robot)

        # motion planner for center of mass
        self.delta = 0.05
        self.max_force = [25, 25]
        self.max_ht = [0.32, 32] # not enforced
        self.k = [[10, 10], [10, 10]]
        self.w = np.array([1e-3, 1e-3, 1e+2, 1e+2, 1, 1, 1e+2, 1e+2])
        self.ter_w = np.array([1e+8, 1e+1, 1e+2, 1e+6, 1, 1])
        self.k_arr = np.tile(self.k,(int(np.round((self.step_time+self.air_time)/self.delta, 2)), 1, 1))

        self.slc_mp = SLCentMotionPlanner(self.delta, 2, self.total_mass, self.inertia, self.max_force, self.max_ht)

    def convert_quat_rpy(self, quat):
        '''
        This function converts quaternion to roll pitch yaw
        Input:
            quat : quaternion to be converted
        '''
        M = se3.XYZQUATToSe3(quat).rotation
        m = sqrt(M[2, 1] ** 2 + M[2, 2] ** 2)
        p = atan2(-M[2, 0], m)

        if abs(abs(p) - np.pi / 2) < 0.001:
            r = 0
            y = -atan2(M[0, 1], M[1, 1])
        else:
            y = atan2(M[1, 0], M[0, 0])  # alpha
            r = atan2(M[2, 1], M[2, 2])  # gamma
        
        lst = [r,p,y]

        return lst

    def get_base_state(self, q, dq):
        '''
        Returns base location and orientation (rpy) at current time step after offset removal
        '''
        com = np.reshape(np.array(q[0:3]), (3,))
        ori = self.convert_quat_rpy(q[3:7])
        base = np.concatenate((com, ori))
        dbase = np.reshape(np.array(dq[0:6]), (6,))

        return np.around(base, 2), np.around(dbase, 2)

    def get_foot_state(self, q, dq):
        '''
        Return foot location after removing the foot diameter
        '''
        fl_foot, fr_foot = self.sse.return_foot_locations(q,dq)
        fl_foot[2] -= self.foot_size
        fr_foot[2] -= self.foot_size

        return fl_foot, fr_foot

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
        fl_foot, fr_foot = self.get_foot_state(q, dq)
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        base, dbase = self.get_base_state(q, dq)
        self.b = fl_hip[1] - fr_hip[1]
        self.n = 1 #right leg on the ground
        self.t = 0

        u = fr_foot
       
        return base[0:3].T, np.around(dbase[0:3].T, 2), np.around(u[0:3], 2), np.power(-1, self.n + 1)

    def generate_traj(self, q, dq, fl_foot, fr_foot, n, u_t, t, des_com, des_dcom):
        '''
        This function returs the desired location of the foot for the 
        given time step
        '''

        x_des = 2*[0.0, 0.0, 0]
        xd_des = 2*[0.0, 0.0, 0]
        q, dq = self.robot.get_state()
            
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        
        if np.power(-1, n) < 0: ## fr leave the ground
            via_point = self.f_lift + u_t[2]
            u_t_des = [[fl_foot[0], fl_foot[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]
            x_des[3:6], xd_des[3:6] = self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],fr_foot[2]], u_t_des[1], [0.0, 0.0, via_point], self.step_time + self.air_time,t)
            x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], des_com[2]])
            
            if t < self.step_time:
                cnt_array = [1, 0]
                x_des[0:3] = np.subtract(u_t_des[0], [fl_hip[0], fl_hip[1], des_com[2]])
                
            elif t < self.step_time + self.air_time:
                cnt_array = [0, 0]
                x_des[0:3] = [-fl_hip[0], -fl_hip[1], -0.1] # fix this

        if np.power(-1, n) > 0: # fl leaves the ground
            via_point = self.f_lift + u_t[2]
            u_t_des = [[u_t[0], u_t[1], u_t[2]], [fr_foot[0], fr_foot[1], fr_foot[2]]]
            x_des[0:3], xd_des[0:3] = self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],fl_foot[2]], u_t_des[0], [0.0, 0.0, via_point], self.step_time + self.air_time,t)
            x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], des_com[2]])
                
            if t < self.step_time:
                cnt_array = [0, 1]
                x_des[3:6] = np.subtract(u_t_des[1], [fr_hip[0], fr_hip[1], des_com[2]])

            elif t < self.step_time + self.air_time:
                cnt_array = [0, 0]
                x_des[3:6] = [-fr_hip[0], -fr_hip[1], -0.1] # fix this

        return x_des, xd_des, cnt_array

    def apply_force(self, F):
        '''
        This function applies a force to the base of the robot
        Input:
            F : force to apply 3d
            
        '''
        pos, ori = p.getBasePositionAndOrientation(self.robot.robotId)
        p.applyExternalForce(objectUniqueId=self.robot.robotId, linkIndex=-1,
                        forceObj=F, posObj=pos, flags=p.WORLD_FRAME)

    def load_terrain(self, dir):
        '''
        This function loads terrain in the simulation
        Input:
            dir : path to urdf
        '''
        terrain = (dir)
        terrain_id = p.loadURDF(terrain)

    def generate_base_traj(self, q, dq, action, n):
        '''
        This function generates the base trajecoty using the motion planner
        Input:
            q: current joint configuration of the robot
            dq : current joint velocities
            action : step location (COP)
            n : index denoting which foot is on the ground
        '''
        base, dbase = self.get_base_state(q, dq)
        x0 = np.concatenate((base, dbase))
        xT = x0.copy()
        xT[2] = action[2] + self.ht
        cnt_plan = [[[0, 0, 0, 0, 0, self.step_time], [0, 0, 0, 0, 0, self.step_time]],
                    [[0, 0, 0, 0, self.step_time, self.step_time + self.air_time], [0, 0, 0, 0, self.step_time, self.step_time + self.air_time]]]

        if np.power(-1, n) < 0: ## fr leave the ground
            cnt_plan[0][0][0] = 1
            cnt_plan[0][0][1:4] = action
        if np.power(-1, n) > 0: # fl leaves the ground
            cnt_plan[0][1][0] = 1
            cnt_plan[0][1][1:4] = action

        traj, force, _ = self.slc_mp.optimize(x0, cnt_plan, self.k_arr, xT, self.w, self.ter_w, self.step_time + self.air_time)

        com = np.repeat(traj[0:3], self.delta/self.dt, axis=1)
        dcom = np.repeat(traj[3:6], self.delta/self.dt, axis=1)
        f = np.repeat(force, self.delta/self.dt, axis=1)

        return com, dcom, f

    def step_env(self, action, force = None):
        '''
        This function simulates the environment for step time duration
        Input:
            action : step location (COP)
        '''
        self.n += 1
        q, dq = self.robot.get_state()
        des_com, des_dcom, f = self.generate_base_traj(q, dq, action, self.n)

        fl_foot, fr_foot = self.get_foot_state(q, dq) ## computing current location of the feet
        u_t = action
        
        for t in range(int((self.step_time+self.air_time)/self.dt)):
            p.stepSimulation()
            time.sleep(0.001)
            if force:
                self.apply_force(force)

            q, dq = self.robot.get_state()

            x_des, xd_des, cnt_array = self.generate_traj(q, dq, fl_foot, fr_foot, \
                            self.n, u_t, self.dt*t, des_com[:,t] - self.base_offset, des_dcom[:,t])

            w_com = self.centr_controller.compute_com_wrench(q, dq, des_com[:,t], des_dcom[:,t],[0, 0, 0, 1], [0, 0, 0])
            w_com[0:3] += f[:,t] # feed forward force
            F = self.centr_controller.compute_force_qp(q, dq, cnt_array, w_com)
            tau = self.bolt_leg_ctrl.return_joint_torques(q,dq,self.kp,self.kd,x_des, xd_des,F)
            self.robot.send_joint_command(tau)
    
            self.t += 1

        q, dq = self.robot.get_state()
        base, dbase = self.get_base_state(q, dq)
        fl_foot, fr_foot = self.get_foot_state(q, dq) ## computing current location of the feet
        if np.power(-1, self.n) < 0:
            u = fr_foot
        else:
            u = fl_foot
        return base[0:3].T, np.around(dbase[0:3].T, 2), np.around(u[0:3], 2), np.power(-1, self.n + 1)

    def start_recording(self, file_name):
        self.robot.start_recording(file_name)

    def stop_recording(self):
        self.robot.stop_recording()