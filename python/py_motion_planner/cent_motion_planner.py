## This file is an implementation of the centroidal motion planner
## that breaks the problem down to make things convex
## Author: Avadesh Meduri
## Date: 1/07/2020

from matplotlib import pyplot as plt
import numpy as np 
from quadprog import solve_qp

from py_motion_planner.sl_cent_motion_planner import SLCentMotionPlanner
from py_motion_planner.cent_fy_optimizer import ForceYOptimizer 
from py_motion_planner.cent_fx_optimizer import ForceXOptimizer
from py_motion_planner.cent_fz_optimizer import ForceZOptimizer

class CentMotionPlanner:

    def __init__(self, dt, max_cnts, mass, inertia, max_force, max_ht):
        '''
        Input:
            dt : time discretization
            max_cnts : total number of contact points (2 for biped)
            mass : mass of the robot
            inertia : inertia [ixx, iyy, izz]
            max_force : maximum force limit in the x,y,z direction
            max_ht : maximum distance of com from the contact location
        '''

        self.dt = dt
        self.g = 9.81
        self.m = mass
        assert len(inertia) == 3
        self.I = inertia

        self.max_cnts = max_cnts

        assert np.shape(max_force) == (self.max_cnts, 3)
        self.max_force = max_force
        assert np.shape(max_ht) == (self.max_cnts, 3)
        self.max_ht = max_ht

        self.fy_optim = ForceYOptimizer(self.dt, self.max_cnts, self.m, self.I[0], self.max_force[:,1], self.max_ht[:,1])
        self.fx_optim = ForceXOptimizer(self.dt, self.max_cnts, self.m, self.I[1], self.max_force[:,0], self.max_ht[:,0])
        self.fz_optim = ForceZOptimizer(self.dt, self.max_cnts, self.m, self.max_force[:,0], self.max_ht[:,0])


    def create_contact_array(self, cnt_plan, no_col):
        '''
        This function creates the contact array [[1/0, 1/0], [..], ...]
        and the contact location array 
        Input:
            cnt_plan : contact plan [1/0, x, y, z, start, end]
            no_col : number of collocation points
        '''
        assert np.shape(cnt_plan)[1] == self.max_cnts
        assert np.shape(cnt_plan)[2] == 6

        cnt_arr = np.zeros((no_col, self.max_cnts))
        r_arr = np.zeros((no_col, self.max_cnts, 3))

        t_arr = np.zeros(self.max_cnts)
        for i in range(len(cnt_plan)):
            for n in range(self.max_cnts):
                time_steps = int(np.round((cnt_plan[i][n][5] - cnt_plan[i][n][4])/self.dt, 2))
                cnt_arr[:,n][int(t_arr[n]):int(t_arr[n]+time_steps)] = cnt_plan[i][n][0]
                r_arr[int(t_arr[n]):int(t_arr[n]+time_steps), n] = np.tile(cnt_plan[i][n][1:4], (time_steps,1))
                t_arr[n] += time_steps

        return cnt_arr, r_arr

    def optimize(self, x0, cnt_plan, xT, w, ter_w, horizon):
        '''
        This function optimizes the motion
        Input:
            x0 : starting state of the robot
            cnt_plan : contact plan [1/0, x, y, z, start, end]
            xT : final desired state
            w : tracking weights
            wT : weight on terminal state
            horizon : duration of motion in seconds
        '''

        no_col = int(np.round(horizon/self.dt, 2))
        
        assert len(x0) == 10
        assert len(xT) == 10
        assert len(w) == 10 + 3*self.max_cnts
        assert len(ter_w) == 10
        
        wx = np.take(w, [0, 3, 7, 9])
        ter_wx = np.take(ter_w, [0, 3, 7, 9])
        wy = np.take(w, [1, 4, 6, 8])
        ter_wy = np.take(ter_w, [1, 4, 6, 8])
        wz = np.take(w, [2, 5])
        ter_wz = np.take(ter_w, [2, 5])

        wx = np.concatenate((wx, w[10::3]))
        wy = np.concatenate((wy, w[11::3]))
        wz = np.concatenate((wz, w[12::3]))
        
        x_init = np.take(x0.copy(), [0, 3, 7, 9])
        y_init = np.take(x0.copy(), [1, 4, 6, 8])
        z_init = np.take(x0.copy(), [2,5])
        
        x_term = np.take(xT.copy(), [0, 3, 7, 9])
        y_term = np.take(xT.copy(), [1, 4, 6, 8])
        z_term = np.take(xT.copy(), [2,5])
        
        cnt_arr, r_arr = self.create_contact_array(cnt_plan, no_col)
        
        traj_z, fz = self.fz_optim.optimize(z_init, cnt_arr, z_term, wz, ter_wz, horizon)
        traj_y, fy = self.fy_optim.optimize(y_init, traj_z[0], fz, cnt_arr, r_arr, y_term, wy, ter_wy, horizon)
        traj_x, fx = self.fx_optim.optimize(x_init, traj_z[0], fz, cnt_arr, r_arr, x_term, wx, ter_wx, horizon)
        
        traj = np.zeros((10, np.shape(traj_z)[1]))
        f = np.zeros((self.max_cnts, 3, np.shape(fz)[1]))
        
        traj[0] = traj_x[0]
        traj[1] = traj_y[0]
        traj[2] = traj_z[0]
        traj[3] = traj_x[1]
        traj[4] = traj_y[1]
        traj[5] = traj_z[1]
        traj[6] = traj_y[2]
        traj[7] = traj_x[2]
        traj[8] = traj_y[3]
        traj[9] = traj_x[3]

        for n in range(self.max_cnts):
            f[n][0] = fx[n]
            f[n][1] = fy[n]
            f[n][2] = fz[n]

        return traj, f