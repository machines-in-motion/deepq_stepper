## This file containts the optimizer to for the spring loaded centeroidal
## dynamics 
## Author : Avadesh Meduri
## Date : 23/06/2020

from matplotlib import pyplot as plt
import numpy as np 
from quadprog import solve_qp

class SLCentMotionPlanner:

    def __init__(self, dt, max_cnts, mass, inertia, max_force, max_ht):
        '''
        Input:
            dt : time discretization
            max_cnts : total number of contact points (2 for biped)
            mass : mass of the robot
            inertia : inertia [ixx, iyy, izz]
            max_force : maximum force limit in the z direction
            max_ht : maximum height the com can go above the contact point
        '''

        self.dt = dt
        self.g = 9.81
        self.m = mass
        assert len(inertia) == 3
        self.I = inertia

        self.max_cnts = max_cnts

        assert len(max_force) == self.max_cnts
        self.max_force = max_force
        assert len(max_ht) == self.max_cnts
        self.max_ht = max_ht

        # initialising the dyn constraint block
        self.dyn_block = np.zeros((6, 12 + self.max_cnts))
        self.dyn_block[0:6, 0:6] = np.identity(6)
        self.dyn_block[0, 1] = self.dt
        self.dyn_block[2, 4] = self.dt
        self.dyn_block[3, 5] = self.dt
        self.dyn_block[0:6,-6:] = -np.identity(6)

        self.dyn_vec = np.zeros(6)
        self.dyn_vec[1] = self.g*self.dt

        # # initialising the inequality constraints
        self.ineq_block = np.zeros((2*self.max_cnts, 6 + self.max_cnts))
        self.ineq_block[0:self.max_cnts,6:] = -np.identity(self.max_cnts)
        self.ineq_block[self.max_cnts:2*self.max_cnts,6:] = np.identity(self.max_cnts)
        
        self.ineq_vec = np.zeros(2*self.max_cnts)
        self.ineq_vec[self.max_cnts:] = self.max_force

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

    def integrate_xy_dynamics(self, xy_t, cnt_t, r_t, k_t):
        '''
        This function integrates xy dynamics by one time step
        Input:
            xy_t : current xy state (x, y, yaw, xd, yd, yaw_d)
            cnt_t : current contact configuration (1/0 for each contact point)
            r_t : current contact locations
            k_t : spring stiffness [[kx, ky], ...]
        '''
        dxy_t = np.zeros(6)
        dxy_t[0:3] = xy_t[3:]
        for n in range(self.max_cnts):
            dxy_t[3] += cnt_t[n]*k_t[n][0]*(xy_t[0] - r_t[n][0]) #fx
            dxy_t[4] += cnt_t[n]*k_t[n][1]*(xy_t[1] - r_t[n][1]) #fy
            dxy_t[5] += cnt_t[n]*(k_t[n][0] - k_t[n][1])*(xy_t[1] - r_t[n][1])*(xy_t[0] - r_t[n][0]) #tz
        
        return np.add(xy_t, dxy_t*self.dt)

    def create_constraint_blocks(self, xy_t, cnt_t, r_t, k_t):
        '''
        This function creates the constraints matrix for one time step
        Input:
            xy_t : xy dynamics at time step t (x, y, yaw, xd, yd, yaw_d)
            cnt_t : current contact configuration (1/0 for each contact point)
            r_t : current contact locations
            k_t : spring stiffness [[kx, ky], ...]
        '''
        # dynamic constraints
        dyn_block = self.dyn_block.copy()
        dyn_vec = self.dyn_vec.copy()
        for n in range(self.max_cnts):
            dyn_block[1,6+n] = cnt_t[n]*self.dt/self.m 

            dyn_block[4,0] += cnt_t[n]*k_t[n][1]*(xy_t[1] - r_t[n][1])*self.dt/self.I[0]
            dyn_block[4,6+n] = cnt_t[n]*(r_t[n][1] - xy_t[1])*self.dt/self.I[0]
            dyn_vec[4] += cnt_t[n]*k_t[n][1]*r_t[n][2]*(xy_t[1] - r_t[n][1])*self.dt/self.I[0]

            dyn_block[5,0] += cnt_t[n]*k_t[n][0]*(r_t[n][0] - xy_t[0])*self.dt/self.I[1]
            dyn_block[5,6+n] = cnt_t[n]*(xy_t[0] - r_t[n][0])*self.dt/self.I[1]
            dyn_vec[5] += cnt_t[n]*k_t[n][0]*r_t[n][2]*(r_t[n][0] - xy_t[0])*self.dt/self.I[1]

        return dyn_block, dyn_vec

    def create_constraints(self, x0, cnt_arr, r_arr, k_arr, no_col):
        '''
        This function creates the equality and inequality constraints for the qp
        Input:
            x0 : starting state of robot
            cnt_arr : contact array
            r_arr : location of contact points during the plan
            k_arr : spring stiffnesss array
            no_cols : number of colocation points
        '''
        self.xy_sim_data = np.zeros((6, no_col+1))
        self.xy_sim_data[:,0] = np.take(x0, [0, 1, 8, 3, 4, 11])
        
        A = np.zeros((6*no_col + 6, (6 + self.max_cnts)*no_col + 6))
        b = np.zeros(6*no_col + 6)

        G = np.zeros((2*self.max_cnts*no_col, (6 + self.max_cnts)*no_col + 6))
        h = np.zeros(2*self.max_cnts*no_col)

        for t in range(no_col):
            self.xy_sim_data[:,t+1] = self.integrate_xy_dynamics(self.xy_sim_data[:,t], cnt_arr[t], r_arr[t], k_arr[t]) 
            dyn_block_t, dyn_vec_t = self.create_constraint_blocks(self.xy_sim_data[:,t], cnt_arr[t], r_arr[t], k_arr[t])    
            A[6*t:6*(t+1), (6+self.max_cnts)*t:(6+self.max_cnts)*(t+1) + 6] = dyn_block_t
            b[6*t:6*(t+1)] = dyn_vec_t

            G[2*self.max_cnts*t:2*self.max_cnts*(t+1), (6+self.max_cnts)*t: (6+self.max_cnts)*(t+1)] = self.ineq_block
            h[2*self.max_cnts*t:2*self.max_cnts*(t+1)] = self.ineq_vec

        # initial constraints
        A[-6:, 0:6] = np.identity(6)
        b[-6:] = np.take(x0, [2,5,6,7,9,10])

        return A, b, G, h

    def create_cost_matrix(self, xT, w, ter_w, no_col):
        '''
        This function creates the cost matrix for the QP
        Input:
            xT : final desired state
            w : weight matrix (cz, czd, thx, thy, thxd, thyd, f, ..)
            ter_w : weight matrix on terminal state
            no_col: number of colocation points
        '''
        assert len(w) == 6 + self.max_cnts
        assert len(ter_w) == 6

        P = np.zeros(((6 + self.max_cnts)*no_col + 6,(6 + self.max_cnts)*no_col + 6))
        q = np.zeros((6 + self.max_cnts)*no_col + 6)

        wt = np.concatenate((np.tile(w, no_col), ter_w))
        np.fill_diagonal(P, wt)
        
        q[-6:] = -ter_w*np.take(xT, [2, 5, 6, 7, 9, 10]) 

        return P, q

    def optimize(self, x0, cnt_plan, k_arr, xT, w, ter_w, horizon):
        '''
        This function optimizes the motion
        Input:
            x0 : starting state of the robot
            cnt_plan : contact plan [1/0, x, y, z, start, end]
            k_arr : stiffness array
            xT : final desired state
            w : tracking weights
            wT : weight on terminal state
            horizon : duration of motion in seconds
        '''

        no_col = int(np.round(horizon/self.dt, 2))
        
        assert len(x0) == 12
        assert np.shape(k_arr) == (no_col, 2, 2)
        
        cnt_arr, r_arr = self.create_contact_array(cnt_plan, no_col)
        P, q = self.create_cost_matrix(xT, w, ter_w, no_col)
        A, b, G, h = self.create_constraints(x0, cnt_arr, r_arr, k_arr, no_col)

        qp_G = P
        qp_a = -q
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]    

        sol = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
        
        traj = np.zeros((12, no_col+1))
        force = np.zeros((self.max_cnts, no_col+1))

        traj[0:2] = self.xy_sim_data[0:2]
        traj[3:5] = self.xy_sim_data[3:5]
        traj[8] = self.xy_sim_data[2]
        traj[11] = self.xy_sim_data[5]
        for t in range(no_col+1):
            traj[2,t] = sol[(6+self.max_cnts)*t] 
            traj[5,t] = sol[(6+self.max_cnts)*t+1]
            traj[6,t] = sol[(6+self.max_cnts)*t+2]
            traj[7,t] = sol[(6+self.max_cnts)*t+3]
            traj[9,t] = sol[(6+self.max_cnts)*t+4]
            traj[10,t] = sol[(6+self.max_cnts)*t+5]
            if t < no_col:
                force[:,t] = sol[(6+self.max_cnts)*t + 6:(6+self.max_cnts)*t + 6 + self.max_cnts]

        return traj, force

    def plot(self, traj, force, horizon):
        
        t = np.linspace(0, horizon, int(np.round(horizon/self.dt, 2)) + 1)

        fig, axs = plt.subplots(4,1)
        axs[0].plot(t, traj[0], label = 'com_x')
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(t, traj[1], label = 'com_y')
        axs[1].grid()
        axs[1].legend()
        axs[2].plot(t, traj[2], label = 'com_z')
        axs[2].grid()
        axs[2].legend()
        for n in range(self.max_cnts):
            axs[3].plot(t, force[n], label = 'f' + str(n))
        axs[3].grid()
        axs[3].legend()
        
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t, traj[3], label = 'com_xd')
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(t, traj[4], label = 'com_yd')
        axs[1].grid()
        axs[1].legend()
        axs[2].plot(t, traj[5], label = 'com_zd')
        axs[2].grid()
        axs[2].legend()
        
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t, traj[6], label = 'com_ang_x')
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(t, traj[7], label = 'com_ang_y')
        axs[1].grid()
        axs[1].legend()
        axs[2].plot(t, traj[8], label = 'com_ang_z')
        axs[2].grid()
        axs[2].legend()
        
        plt.show()


            
