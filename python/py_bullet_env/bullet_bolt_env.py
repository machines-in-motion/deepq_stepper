## Implementation of botl bullet environment to test the 
## Deep Q stepper
## Author : Avadesh Meduri
## Date : 4/6/2020


import numpy as np
from math import atan2, pi, sqrt

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

from py_motion_planner.ip_motion_planner import IPMotionPlanner

from pinocchio.utils import zero

from matplotlib import pyplot as plt

class BoltBulletEnv:

    def __init__(self, ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w = None):
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
            w : weights to compute cost
        '''

        self.ht = ht
        self.step_time = step_time
        self.stance_time = stance_time
        self.dt = 0.001
        
        self.robot = BoltRobot()
        self.total_mass = sum([i.mass for i in self.robot.pin_robot.model.inertias[1:]])
        # Robot parameters to account for to match IPM training env
        # size of the foot diameter
        self.foot_size = 0.02
        # This is the distance of the hip from the true COM
        # This is subtracted as in the training the COM is assumed to be
        # at the hip joint with no offset 
        # Impedance controller iniitialization
        self.bolt_leg_ctrl = BoltImpedanceController(self.robot)

        self.b = 0.13

        # Centroidal controller initialisation
        assert np.shape(kp) == (3,)
        assert np.shape(kd) == (3,) 

        self.kp = 2 * kp
        self.kd = 2 * kd

        self.centr_controller = BoltCentroidalController(self.robot.pin_robot, self.total_mass,
        mu=1.0, kp=kp_com, kd=kd_com, kpa=kp_ang_com, kda=kd_ang_com,
        eff_ids=self.robot.pinocchio_endeff_ids)
        self.F = np.zeros(6)

        # Trajectory Generator initialisation
        self.trj = TrajGenerator(self.robot.pin_robot)
        self.f_lift = 0.06 ## height the foot lifts of the ground
        
        # State estimation initialisation
        self.sse = BoltStateEstimator(self.robot.pin_robot)

        # motion planner for center of mass
        self.delta = 0.01
        self.max_acc = 8
        self.ip = IPMotionPlanner(self.delta, self.max_acc)


        self.w = w

        ## arrays to store data
        self.act_com = []
        self.act_dcom = []
        self.des_com = []
        self.des_dcom = []
        self.act_leg_length = []
        self.des_leg_length = []


    def get_com_state(self, q, dq):
        '''
        Returns center of mass location at current time step after offset removal
        '''
        com = np.reshape(np.array(q[0:3]), (3,))
        dcom = np.reshape(np.array(dq[0:3]), (3,))

        return np.around(com, 2), np.around(dcom, 2)

    def get_foot_state(self, q, dq):
        '''
        Return foot location after removing the foot diameter
        '''
        fl_foot, fr_foot = self.sse.return_foot_locations(q,dq)
        fl_foot[2] -= self.foot_size
        fr_foot[2] -= self.foot_size

        return fl_foot, fr_foot

    def convert_quat_rpy(self, quat):
        '''
        This function converts quaternion to roll pitch yaw
        Input:
            quat : quaternion to be converted
        '''
        M = se3.XYZQUATToSE3(quat).rotation
        m = sqrt(M[2, 1] ** 2 + M[2, 2] ** 2)
        p = atan2(-M[2, 0], m)

        if abs(abs(p) - np.pi / 2) < 0.001:
            r = 0
            y = -atan2(M[0, 1], M[1, 1])
        else:
            y = atan2(M[1, 0], M[0, 0])  # alpha
            r = atan2(M[2, 1], M[2, 2])  # gamma
        
        lst = np.array([r,p,y])

        return lst

    def update_gains(self, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com):
        self.kp = 2 * kp
        self.kd = 2 * kd
        self.centr_controller = BoltCentroidalController(self.robot.pin_robot, self.total_mass,
        mu=0.5, kp=kp_com, kd=kd_com, kpa=kp_ang_com, kda=kd_ang_com,
        eff_ids=self.robot.pinocchio_endeff_ids)
        
    def reset_env(self, state = None):
        '''
        This function resets the environment
        Input:
            state : initial state of robot [x,y,z, xd,yd]
        '''
        q0 = np.matrix(BoltConfig.initial_configuration).T
        dq0 = np.matrix(BoltConfig.initial_velocity).T
        
        if state:
            q0[0:3] = np.array([state[0:3]]).T
            dq0[0:2] = np.array([state[3:5]]).T
        
        self.robot.reset_state(q0, dq0)

        self.xd_des = 2*[0,0,0]
        
        self.x_ori = [0., 0., 0., 1.]
        self.x_angvel = [0., 0., 0.]
        
        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.get_foot_state(q, dq)
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        com, dcom = self.get_com_state(q, dq)
        
        self.b = fl_hip[1] - fr_hip[1]
        self.n = 1 #right leg on the ground
        self.t = 0

        self.u = fr_foot
       
        return com[0:3].T, np.around(dcom[0:3].T, 2), np.around(self.u[0:3], 2), np.power(-1, self.n + 1)

    def generate_traj(self, q, dq, fl_foot, fr_foot, n, u_t, t, stance_time, des_z, des_zd):
        '''
        This function returs the desired location of the foot for the 
        given time step
        '''

        x_des = 2*[0.0, 0.0, 0]
        xd_des = 2*[0.0, 0.0, 0]
        com, dcom = self.get_com_state(q, dq)
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        
        des_z -= com[2] - 0.5*(fl_hip[2] + fr_hip[2])
        
        if t < self.step_time - stance_time:
            if np.power(-1, n) < 0: ## fr leave the ground
                cnt_array = [1, 0]
                via_point = self.f_lift + u_t[2]
                u_t_des = [[fl_foot[0], fl_foot[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]
                x_des[0:3] = np.subtract(u_t_des[0], [fl_hip[0], fl_hip[1], des_z])
                # xd_des[0:3] = [com[0], com[1], des_zd]
                x_des[3:6], xd_des[3:6] = self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],fr_foot[2]], u_t_des[1], [0.0, 0.0, via_point], self.step_time - stance_time,t)
                x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], des_z])
                
            if np.power(-1, n) > 0: # fl leaves the ground
                cnt_array = [0, 1]
                via_point = self.f_lift + u_t[2]
                u_t_des = [[u_t[0], u_t[1], u_t[2]], [fr_foot[0], fr_foot[1], fr_foot[2]]]
                x_des[0:3], xd_des[0:3] = self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],fl_foot[2]], u_t_des[0], [0.0, 0.0, via_point], self.step_time - stance_time,t)
                x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], des_z])
                x_des[3:6] = np.subtract(u_t_des[1], [fr_hip[0], fr_hip[1], des_z])
                # xd_des[3:6] = [com[0], com[1], des_zd]

        else:
            cnt_array = [1, 1]
            if np.power(-1, n) < 0:
                u_t_des = [[fl_foot[0], fl_foot[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]
            if np.power(-1, n) > 0:
                u_t_des = [[u_t[0], u_t[1], u_t[2]], [fr_foot[0], fr_foot[1], fr_foot[2]]]
            
            x_des[0:3] = np.subtract(u_t_des[0], [fl_hip[0], fl_hip[1], des_z])
            x_des[3:6] = np.subtract(u_t_des[1], [fr_hip[0], fr_hip[1], des_z])
            # xd_des[0:3] = [com[0], com[1], 0]
            # xd_des[3:6] = [com[0], com[1],0]


        self.des_leg_length.append(x_des[2])
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

    def step_env(self, action, des_vel, force = None):
        '''
        This function simulates the environment for step time duration
        Input:
            action : step location (COP)
        '''
        self.n += 1
        done = False
        q, dq = self.robot.get_state()
        com, dcom = self.get_com_state(q, dq)
        des_z, des_zd, _ = self.ip.generate_force_trajectory(com[2], action[2], self.step_time, self.ht)
        des_z = np.repeat(des_z, self.delta/self.dt)
        des_zd = np.repeat(des_zd, self.delta/self.dt)
        self.des_com.append(des_z[:int(self.step_time/self.dt)])
        fl_foot, fr_foot = self.get_foot_state(q, dq) ## computing current location of the feet
        u_t = action
        
        for t in range(int(self.step_time/self.dt)):
            p.stepSimulation()
            # time.sleep(0.001)
            self.apply_force(force)
            q, dq = self.robot.get_state()

            self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_dcom.append(np.reshape(np.array(dq[0:3]), (3,)))
            self.des_dcom.append(des_vel[0:2])

            x_des, xd_des, cnt_array = self.generate_traj(q, dq, fl_foot, fr_foot, \
                            self.n, u_t, self.dt*t, self.stance_time, des_z[t] - self.foot_size, des_zd[t])
            
            des_com = [0, 0, des_z[t]]
            des_vel = [des_vel[0], des_vel[1], des_zd[t]]
            w_com = self.centr_controller.compute_com_wrench(q, dq, des_com, des_vel,[0, 0, 0, 1], [0, 0, 0])
            w_com[2] += 0.85*self.total_mass * 9.81
            F = self.centr_controller.compute_force_qp(q, dq, cnt_array, w_com)
            tau = self.bolt_leg_ctrl.return_joint_torques(q,dq,self.kp,self.kd,x_des, xd_des,F)
            self.robot.send_joint_command(tau)
            
            if not done: 
                if self.is_done():
                    done = True

            self.t += 1
        
        com, dcom = self.get_com_state(q, dq)
        fl_foot, fr_foot = self.get_foot_state(q, dq) ## computing current location of the feet
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        if np.power(-1, self.n) < 0:
            cost = self.compute_cost(com, dcom, fl_hip, fr_hip, self.n, fr_foot, self.u, des_vel, done)
            self.u = fr_foot
        else:
            cost = self.compute_cost(com, dcom, fl_hip, fr_hip, self.n, fl_foot, self.u, des_vel, done)
            self.u = fl_foot
        
        return com[0:3].T, np.around(dcom[0:3].T, 2), np.around(self.u[0:3], 2), np.power(-1, self.n + 1), cost, done

    def is_done(self):
        '''
        This function checks if the episode should be terminated.
        '''
        q, dq = self.robot.get_state()
        com, dcom = self.get_com_state(q, dq)
        ori = abs((180/np.pi)*self.convert_quat_rpy(q[3:7]))
        fl_foot, fr_foot = self.sse.return_foot_locations(q,dq)
        fl_hip, fr_hip = self.sse.return_hip_locations(q, dq)
        self.act_leg_length.append(fl_foot[2] - fl_hip[2])
        if com[2] < 0.1:
            # print('ht')
            return True
        elif np.linalg.norm(dcom,2) > 2:
            # print('vel')
            return True
        elif ori[0] > 30 or ori[1] > 30 or ori[2] > 20:
            # print('ori')
            return True
        elif np.linalg.norm(fl_foot - fl_hip) > 0.31 or np.linalg.norm(fr_foot - fr_hip) > 0.31:
            # print('leg', np.linalg.norm(fl_foot - fl_hip), np.linalg.norm(fr_foot - fr_hip))
            return True
        elif np.linalg.norm(fl_foot - fr_foot) < 0.02:
            # print('foot collision')
            return True
        else:
            return False
    
    def compute_cost(self, com, dcom, fl_hip, fr_hip, n, u, u_old, v_des, done):
        '''
        Computes cost which is distance between the hip(closest hip depending on which foot is on the ground)
        and the foot + velocity of the center of mass + 1 if step length not equal to zero (after taking into
        account the offset) + 100 if episode terminates (kinematics constraints are violated) 
        '''

        if np.power(-1, self.n) < 0:
            cost = self.w[0]*(abs(fr_hip[0] - u[0]) + abs(fr_hip[1] - u[1]))
        else:
            cost = self.w[0]*(abs(fl_hip[0] - u[0]) + abs(fl_hip[1] - u[1]))

        cost += self.w[1]*(abs(v_des[0] - dcom[0]) + abs(v_des[1] - dcom[1]))
        cost += self.w[2]*(abs(u[0] - u_old[0]) + abs(abs(u[1] - u_old[1]) - self.b) + abs(u[2] - u_old[2]))

        if done:
            cost += 100

        return np.round(cost, 2) 
        

    def start_recording(self, file_name):
        self.robot.start_recording(file_name)

    def stop_recording(self):
        self.robot.stop_recording()

    def plot(self):
        
        self.act_com = np.asarray(self.act_com)
        self.act_dcom = np.asarray(self.act_dcom)
        self.des_dcom = np.asarray(self.des_dcom)
        self.des_com = np.reshape(self.des_com, (len(self.act_com[:,0]), ))

        T = len(self.act_com[:,0])
        t = 0.001*np.arange(0,T)

        fig, ax = plt.subplots(6,1)
        ax[0].plot(t,self.act_com[:,0], label = 'cx')
        ax[0].grid()
        ax[0].legend()
        ax[0].set_ylabel('meters')

        ax[1].plot(t,self.act_com[:,1], label = 'cy')
        ax[1].grid()
        ax[1].legend()
        ax[1].set_ylabel('meters')

        ax[2].plot(t,self.act_com[:,2], label = 'cz')
        ax[2].plot(t,self.des_com, label = 'des_cz')
        ax[2].grid()
        ax[2].legend()
        ax[2].set_ylabel('meters')

        ax[3].plot(t,self.act_dcom[:,0], label = 'vx')
        ax[3].plot(t,self.des_dcom[:,0], label = 'des_vx')
        ax[3].grid()
        ax[3].legend()
        ax[3].set_ylabel('meters/second')

        ax[4].plot(t,self.act_dcom[:,1], label = 'vy')
        ax[4].plot(t,self.des_dcom[:,1], label = 'des_vy')
        ax[4].grid()
        ax[4].legend()
        ax[4].set_ylabel('meters/second')

        ax[5].plot(t,self.act_dcom[:,2], label = 'vz')
        ax[5].grid()
        ax[5].legend()
        ax[5].set_ylabel('meters/second')
    
        fig, ax1 = plt.subplots(2,1)
        ax1[0].plot(self.act_leg_length, label = 'act')
        ax1[0].plot(self.des_leg_length, label = 'des')
        ax1[0].grid()
        ax1[0].legend()
        ax1[0].set_ylabel('meters')

        plt.show()