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

from py_motion_planner.cent_motion_planner import CentMotionPlanner

from pinocchio.utils import zero

from matplotlib import pyplot as plt


class BulletCentBoltEnv:

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
        self.inertia = np.diag(self.robot.pin_robot.mass(q0)[3:6, 3:6]).copy()
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
        #this to generate a trajectory for the foot lifting of the ground through the air face
        self.trj_lift = TrajGenerator(self.robot.pin_robot)
        self.f_lift = 0.06 ## height the foot lifts of the ground
        
        # State estimation initialisation
        self.sse = BoltStateEstimator(self.robot.pin_robot)

        # motion planner for center of mass
        self.delta_t = 0.025
        self.f_max = np.array([[30,30, 30], [30, 30, 30]])
        self.max_ht = np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]])
        self.w = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e+2, 1e+2, 1e+7, 1e+7, 1e-4, 1e-4, 1e+1, 1e-4, 1e-4, 1e+1])
        self.ter_w = np.array([1e-4, 1e-4, 1e+8, 1e-4, 1e-4, 1e+5, 1e+3, 1e+3, 1e+6, 1e+6])
        self.xt = [0, 0, self.ht, 0, 0, 0, 0, 0, 0, 0]

        self.cent_mp = CentMotionPlanner(self.delta_t, 2, self.total_mass, self.inertia, self.f_max, self.max_ht)
        
        ## arrays to store data

        self.des_com = []
        self.act_com = []
        self.act_ori = []

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
        ori = self.convert_quat_rpy(q[3:7])[0:2] #removing yaw
        base = np.concatenate((com, np.reshape(np.array(dq[0:3]), (3,))))
        base_ori = np.concatenate((ori, np.reshape(np.array(dq[3:5]), (2,))))

        return np.around(base, 2), np.around(base_ori, 2)

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
        base, base_ori = self.get_base_state(q, dq)
        self.b = fl_hip[1] - fr_hip[1]
        self.n = 1 #right leg on the ground
        self.t = 0

        self.u = np.around(fr_foot[0:3], 2)
       
        return base[0:5].T, self.u, np.power(-1, self.n + 1)

    def generate_foot_traj(self, q, dq, fl_foot, fr_foot, n, u_t, t, des_com, des_dcom):
        '''
        This function returs the desired location of the foot for the 
        given time stepself.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))
        '''

        x_des = 2*[0.0,self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))
        xd_des = 2*[0.0self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))
        q, dq = self.roself.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))te()
            
        fl_hip, fr_hip self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))return_hip_locations(q, dq)
        
        if np.power(-1, n) < 0: ## fr reaches the ground
            via_point = self.f_lift + u_t[2]
            u_t_des = [[fl_foot[0], fl_foot[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]
            
            if t < self.step_time:
                cnt_array = [1, 0]
                u_t_des = [[fl_foot[0], fl_foot[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]

                x_des[0:3] = np.subtract(u_t_des[0], [fl_hip[0], fl_hip[1], des_com[2]])
                xd_des[0:3] = np.subtract([0, 0, 0], des_dcom)
                x_des[3:6], xd_des[3:6] = self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],fr_foot[2]], u_t_des[1], [0.0, 0.0, via_point], self.step_time + self.air_time,t)
                x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], des_com[2]])
                xd_des[3:6] = np.subtract(xd_des[3:6], des_dcom)

            elif t < self.step_time + self.air_time:
                cnt_array = [0, 0]

                u_t_des = [[fl_hip[0], fl_hip[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]
                u_t_des[0][2] = fl_foot[2] + self.f_lift
                x_des[0:3], xd_des[0:3] = self.trj_lift.generate_foot_traj([fl_foot[0], fl_foot[1],fl_foot[2]], u_t_des[0], [0.0, 0.0, 0.5*u_t_des[0][2]], self.step_time + self.air_time,t-self.step_time)
                x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], des_com[2]])
                xd_des[0:3] = np.subtract(xd_des[0:3], des_dcom)

                x_des[3:6], xd_des[3:6] = self.trj.generate_foot_traj([fr_foot[0], fr_foot[1],fr_foot[2]], u_t_des[1], [0.0, 0.0, via_point], self.step_time + self.air_time,t)
                x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], des_com[2]])
                xd_des[3:6] = np.subtract(xd_des[3:6], des_dcom)

            elif t < 2*self.step_time + self.air_time: # fr on ground fl in the air
                cnt_array = [0, 1]

                u_t_des = [[fl_hip[0], fl_hip[1], fl_foot[2]], [u_t[0], u_t[1], u_t[2]]]
                u_t_des[0][2] = fl_foot[2] + self.f_lift

                x_des[0:3], xd_des[0:3] = self.trj_lift.generate_foot_traj([fl_foot[0], fl_foot[1],fl_foot[2]], u_t_des[0], [0.0, 0.0, 0.5*u_t_des[0][2]], self.step_time + self.air_time,t-self.step_time)
                x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], des_com[2]])
                xd_des[0:3] = np.subtract(xd_des[0:3], des_dcom)
                
                x_des[3:6] = np.subtract(u_t_des[1], [fr_hip[0], fr_hip[1], des_com[2]])
                xd_des[3:6] = np.subtract([0, 0, 0], des_dcom)
                
        if np.power(-1, n) > 0: # fl reaches the ground
            via_point = self.f_lift + u_t[2]

            if t < self.step_time: #fl in the air
                cnt_array = [0, 1]
                u_t_des = [[u_t[0], u_t[1], u_t[2]], [fr_foot[0], fr_foot[1], fr_foot[2]]]

                x_des[0:3], xd_des[0:3] = self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],fl_foot[2]], u_t_des[0], [0.0, 0.0, via_point], self.step_time + self.air_time,t)
                x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], des_com[2]])
                xd_des[0:3] = np.subtract(xd_des[0:3], des_dcom)
                x_des[3:6] = np.subtract(u_t_des[1], [fr_hip[0], fr_hip[1], des_com[2]])
                xd_des[3:6] = np.subtract([0, 0, 0], des_dcom)
    
            elif t < self.step_time + self.air_time: # both feet in the air
                cnt_array = [0, 0]
                u_t_des = [[u_t[0], u_t[1], u_t[2]], [fr_hip[0], fr_hip[1], fr_foot[2] + self.f_lift]]
                x_des[0:3], xd_des[0:3] = self.trj.generate_foot_traj([fl_foot[0], fl_foot[1],fl_foot[2]], u_t_des[0], [0.0, 0.0, via_point], self.step_time + self.air_time,t)
                x_des[0:3] = np.subtract(x_des[0:3], [fl_hip[0], fl_hip[1], des_com[2]])
                xd_des[0:3] = np.subtract(xd_des[0:3], des_dcom)
                
                x_des[3:6], xd_des[3:6] = self.trj_lift.generate_foot_traj([fr_foot[0], fr_foot[1],fr_foot[2]], u_t_des[1], [0.0, 0.0, 0.5*u_t_des[1][2]], self.step_time + self.air_time,t - self.step_time)
                x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], des_com[2]])
                xd_des[3:6] = np.subtract(xd_des[3:6], des_dcom)
                
            elif t < 2*self.step_time + self.air_time: # fl on ground fr in the air
                cnt_array = [1, 0]
                u_t_des = [[u_t[0], u_t[1], u_t[2]], [fr_hip[0], fr_hip[1], fr_foot[2] + self.f_lift]]
                x_des[0:3] = np.subtract(u_t_des[0], [fl_hip[0], fl_hip[1], des_com[2]])
                xd_des[0:3] = np.subtract([0, 0, 0], des_dcom)

                x_des[3:6], xd_des[3:6] = self.trj_lift.generate_foot_traj([fr_foot[0], fr_foot[1],fr_foot[2]], u_t_des[1], [0.0, 0.0, 0.5*u_t_des[1][2]], self.step_time + self.air_time,t - self.step_time)
                x_des[3:6] = np.subtract(x_des[3:6], [fr_hip[0], fr_hip[1], des_com[2]])
                xd_des[3:6] = np.subtract(xd_des[3:6], des_dcom)
                
        return x_des, xd_des, cnt_array

    def apply_force(self, F):
        '''self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))
        Thiself.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7])) the base of the robot
        Inpself.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))
           self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))
            
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

    def generate_base_traj(self, q, dq, u, action, n):
        '''
        This function generates the base trajecoty using the motion planner
        Input:
            q: current joint configuration of the robot
            dq : current joint velocities
            u : current foot location (Cop)
            action : step location (COP)
            n : index denoting which foot is on the ground
        '''
        base, base_ori = self.get_base_state(q, dq)
        x0 = np.concatenate((base, base_ori))
        xT = x0.copy()
        xT[2] = action[2] + self.ht
        cnt_plan = [[[0, 0, 0, 0, 0, self.step_time], [0, 0, 0, 0, 0, self.step_time]],
                    [[0, 0, 0, 0, self.step_time, self.step_time + self.air_time], [0, 0, 0, 0, self.step_time, self.step_time + self.air_time]],
                    [[0, 0, 0, 0, self.step_time + self.air_time, np.round(2*self.step_time + self.air_time,2)], [0, 0, 0, 0, self.step_time + self.air_time, np.round(2*self.step_time + self.air_time,2)]]]

        if np.power(-1, n) < 0: ## fr lands on the ground
            cnt_plan[0][0][0] = 1
            cnt_plan[0][0][1:4] = u
            
            cnt_plan[2][1][0] = 1
            cnt_plan[2][1][1:4] = action
        
        if np.power(-1, n) > 0: # fl lands on the ground
            cnt_plan[0][1][0] = 1
            cnt_plan[0][1][1:4] = u
        
            cnt_plan[2][0][0] = 1
            cnt_plan[2][0][1:4] = action
        
        traj, force = self.cent_mp.optimize(x0, cnt_plan, xT, self.w, self.ter_w, np.round(2*self.step_time + self.air_time,2))
        
        com = np.repeat(traj[0:3], self.delta_t/self.dt, axis=1)
        dcom = np.repeat(traj[3:6], self.delta_t/self.dt, axis=1)
        
        f = np.zeros(np.shape(force)[1:])
        for n in range(2):
            f[0] += force[n][0]
            f[1] += force[n][1]
            f[2] += force[n][2]
            
        f = np.repeat(f, self.delta_t/self.dt, axis=1)

        return com, dcom, f

    def step_env(self, action, force = None):
        '''
        This function simulates the environment for step time duration
        Input:
            action : step location (COP)
        '''
        self.n += 1
        q, dq = self.robot.get_state()

        des_com, des_dcom, f = self.generate_base_traj(q, dq, self.u, action, self.n)

        if len(self.des_com) < 2:
            self.des_com = des_com
        else:
            self.des_com = np.concatenate((self.des_com, des_com), axis = 1)

        fl_foot, fr_foot = self.get_foot_state(q, dq) ## computing current location of the feet
        u_t = action
        
        for t in range(int((2*self.step_time+self.air_time)/self.dt)):
            p.stepSimulation()
            time.sleep(0.0005)
            if force:
                self.apply_force(force)

            q, dq = self.robot.get_state()

            self.act_com.append(np.reshape(np.array(q[0:3]), (3,)))
            self.act_ori.append(self.convert_quat_rpy(q[3:7]))

            x_des, xd_des, cnt_array = self.generate_foot_traj(q, dq, fl_foot, fr_foot, \
                            self.n, u_t, self.dt*t, des_com[:,t] - self.base_offset, des_dcom[:,t])
            
            w_com = self.centr_controller.compute_com_wrench(q, dq, des_com[:,t], des_dcom[:,t],[0, 0, 0, 1], [0, 0, 0])
            w_com[0:3] += f[:,t] # feed forward force
            F = self.centr_controller.compute_force_qp(q, dq, cnt_array, w_com)
            tau = self.bolt_leg_ctrl.return_joint_torques(q,dq,self.kp,self.kd,x_des, xd_des,F)

            self.robot.send_joint_command(tau)
    
            self.t += 1

        q, dq = self.robot.get_state()
        fl_foot, fr_foot = self.get_foot_state(q, dq) ## computing current location of the feet
        base, base_ori = self.get_base_state(q, dq)

        if np.power(-1, self.n) < 0:
            self.u = np.around(fr_foot[0:3], 2)
        else:
            self.u = np.around(fl_foot[0:3], 2)

        return base[0:5].T, self.u, np.power(-1, self.n + 1)

    def start_recording(self, file_name):
        self.robot.start_recording(file_name)

    def stop_recording(self):
        self.robot.stop_recording()

    def plot(self):

        self.act_com = np.asarray(self.act_com)
        self.act_ori = np.asarray(self.act_ori)
        self.des_com = np.asarray(self.des_com)
        T = len(self.act_com[:,0])
        t = 0.001*np.arange(0,T)

        fig, ax = plt.subplots(6,1)
        ax[0].plot(t,self.act_com[:,0], label = 'cx')
        ax[0].plot(t, self.des_com[0][0:T], label = 'des_cx')
        ax[0].grid()
        ax[0].legend()
        ax[0].set_ylabel('meters')

        ax[1].plot(t,self.act_com[:,1], label = 'cy')
        ax[1].plot(t, self.des_com[1][0:T], label = 'des_cy')
        ax[1].grid()
        ax[1].legend()
        ax[1].set_ylabel('meters')

        ax[2].plot(t,self.act_com[:,2], label = 'cz')
        ax[2].plot(t, self.des_com[2][0:T], label = 'des_cz')
        ax[2].grid()
        ax[2].legend()
        ax[2].set_ylabel('meters')


        ax[3].plot(t,self.act_ori[:,0], label = 'ang_cx')
        ax[3].plot(t, np.zeros(T), label = 'des_ang_cx')
        ax[3].grid()
        ax[3].legend()
        ax[3].set_ylabel('degree')
        
        ax[4].plot(t,self.act_ori[:,1], label = 'ang_cy')
        ax[4].plot(t, np.zeros(T), label = 'des_ang_cx')
        ax[4].grid()
        ax[4].legend()
        ax[4].set_ylabel('degree')

        ax[5].plot(t,self.act_ori[:,2], label = 'ang_cz')
        ax[5].plot(t, np.zeros(T), label = 'des_ang_cz')
        ax[5].grid()
        ax[5].legend()
        ax[5].set_ylabel('degree')
        ax[5].set_xlabel('sec')

        plt.show()