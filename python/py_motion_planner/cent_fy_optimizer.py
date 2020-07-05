## This file is an implementation of motion planner that 
## regulates the force in the y direction and roll
## Author: Avadesh Meduri
## Date : 1/07/2020

from matplotlib import pyplot as plt
import numpy as np 
from quadprog import solve_qp

class ForceYOptimizer:

    def __init__(self, dt, max_cnts, mass, inertia_x, max_force, max_ht):
        '''
        Input:
            dt : time discretization
            max_cnts : total number of contact points (2 for biped)
            mass : mass of the robot
            inertia_x : inertia about the x axis
            max_force : maximum force limit in the y direction
            max_ht : maximum distance between the COM and contact point in y
        '''

        self.dt = dt
        self.g = 9.81
        self.m = mass
        self.Ixx = inertia_x

        self.max_cnts = max_cnts

        assert len(max_force) == self.max_cnts
        self.max_force = max_force
        assert len(max_ht) == self.max_cnts
        self.max_ht = max_ht

        self.dyn_block = np.zeros((4, 8 + self.max_cnts))
        self.dyn_block[0:4,0:4] = np.identity(4)
        self.dyn_block[0, 1] = self.dt
        self.dyn_block[2, 3] = self.dt
        self.dyn_block[-4:, -4:] = -np.identity(4)

        self.dyn_vec = np.zeros(4)

        #inequalities block
        self.ineq_block = np.zeros((2*self.max_cnts, 4 +self.max_cnts))
        self.ineq_vec = np.zeros(2*self.max_cnts)
        # for n in range(self.max_cnts):
        #     self.ineq_block[2*n, 4 + n] = 1
        #     self.ineq_block[2*n+1, 4 + n] = -1
        #     self.ineq_vec[2*n] = self.max_force[n]
        #     self.ineq_vec[2*n+1] = self.max_force[n]
        

    def create_constraint_blocks(self, cnt_t, r_t, cz_t, fz_t):
        '''
        This function creates the constraints matrix for one time step
        Input:
            cnt_t : current contact configuration (1/0 for each contact point)
            r_t : current contact locations
            cz_t : center of mass location in z
            fz_t : force in the z direction of all contact points
        '''
        # dynamic constraints
        dyn_block = self.dyn_block.copy()
        dyn_vec = self.dyn_vec.copy()

        for n in range(self.max_cnts):
            dyn_block[1,4+n] = cnt_t[n]*self.dt/self.m
            
            dyn_block[3, 0] -= cnt_t[n]*(fz_t[n])*self.dt/self.Ixx
            dyn_block[3, 4+n] = cnt_t[n]*(cz_t - r_t[n,2])*self.dt/self.Ixx
            dyn_vec[3] -= cnt_t[n]*fz_t[n]*r_t[n,1]*self.dt/self.Ixx

        return dyn_block, dyn_vec


    def create_constraints(self, y0, traj_z, fz, cnt_arr, r_arr, no_col):
        '''
        This function creates the equality and inequality constraints for the qp
        Input:
            y0: initial position (y, yd, roll, omega_x)
            traj_z : trajectory from the z optimizer cz
            traj_z : fz for each contact point
            cnt_arr : contact array
            r_arr : location of contact points during the plan
            k_arr : spring stiffnesss array
            no_cols : number of colocation points
        '''

        A = np.zeros((4*no_col + 4, (4 + self.max_cnts)*no_col + 4))
        b = np.zeros(4*no_col + 4)

        G = np.zeros((2*self.max_cnts*no_col, (4 + self.max_cnts)*no_col + 4))
        h = np.zeros(2*self.max_cnts*no_col)

        for t in range(no_col):
            dyn_block_t, dyn_vec_t = self.create_constraint_blocks(cnt_arr[t], r_arr[t], traj_z[t], fz[:,t])
            A[4*t:4*(t+1), (4+self.max_cnts)*t:(4+self.max_cnts)*(t+1) + 4] = dyn_block_t
            b[4*t:4*(t+1)] = dyn_vec_t

            G[2*self.max_cnts*t:2*self.max_cnts*(t+1), (4+self.max_cnts)*t:(4+self.max_cnts)*(t+1)] = self.ineq_block
            h[2*self.max_cnts*t:2*self.max_cnts*(t+1)] = self.ineq_vec


        A[-4:, 0:4] = np.identity(4)
        b[-4:] = y0

        return A, b, G, h

    def create_cost_matrix(self, yT, w, ter_w, no_col):
        '''
        This function creates the cost matrix for the QP
        Input:
            yT : final desired state [y, yd, thx, dthx]
            w : weight matrix (cz, czd, thx, thy, thxd, thyd, f, ..)
            ter_w : weight matrix on terminal state
            no_col: number of colocation points
        '''
    
        P = np.zeros(((4 + self.max_cnts)*no_col + 4,(4 + self.max_cnts)*no_col + 4))
        q = np.zeros((4 + self.max_cnts)*no_col + 4)

        wt = np.concatenate((np.tile(w, no_col), ter_w))
        np.fill_diagonal(P, wt)
        
        q[-4:] = -ter_w*yT 

        return P, q

    def optimize(self, y0, traj_z, fz, cnt_arr, r_arr, yT, w, ter_w, horizon):
        '''
        This function optimizes the motion
        Input:
            y0 : starting state of the robot
            traj_z : trajectory of comz 
            fz : trajectory fz
            cnt_arr : contact array
            r_arr : location of contact points during the plan
            yT : final desired state
            w : tracking weights
            wT : weight on terminal state
            horizon : duration of motion in seconds
        '''

        no_col = int(np.round(horizon/self.dt, 2))
        
        assert len(w) == 4 + self.max_cnts
        assert len(ter_w) == 4

        assert len(y0) == 4
        assert len(traj_z) == no_col+1
        assert np.shape(fz) == (self.max_cnts, no_col+1)


        A, b, G, h = self.create_constraints(y0, traj_z, fz, cnt_arr, r_arr, no_col)
        P, q = self.create_cost_matrix(yT, w, ter_w, no_col)

        qp_G = P
        qp_a = -q
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]    

        sol = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
        
        traj_y = np.zeros((4, no_col + 1))
        fy = np.zeros((self.max_cnts, no_col+1))

        for t in range(no_col+1):
            traj_y[0, t] = sol[t*(4+self.max_cnts)]
            traj_y[1, t] = sol[t*(4+self.max_cnts) + 1]
            traj_y[2, t] = sol[t*(4+self.max_cnts) + 2]
            traj_y[3, t] = sol[t*(4+self.max_cnts) + 3]
            if t < no_col:
                fy[:,t] = sol[(4+self.max_cnts)*t + 4:(4+self.max_cnts)*t + 4 + self.max_cnts]
            
        return traj_y, fy
