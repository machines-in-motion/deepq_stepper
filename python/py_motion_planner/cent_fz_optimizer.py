## This file generates a z direction trajectory of the robot
## Author : Avadesh Meduri
## Date : 3/07/2020

from matplotlib import pyplot as plt
import numpy as np 
from quadprog import solve_qp

class ForceZOptimizer:

    def __init__(self,  dt, max_cnts, mass, max_force, max_ht):
        '''
        Input:
            dt : time discretization
            max_cnts : total number of contact points (2 for biped)
            mass : mass of the robot
            max_force : maximum force limit in the z direction
            max_ht : maximum distance between the COM and contact point in z
        '''

        self.dt = dt
        self.g = 9.81
        self.m = mass

        self.max_cnts = max_cnts

        assert len(max_force) == self.max_cnts
        self.max_force = max_force
        assert len(max_ht) == self.max_cnts
        self.max_ht = max_ht

        #dynamic constraints
        self.dyn_block = np.zeros((2, 4 + self.max_cnts))
        self.dyn_block[0:2, 0:2] = np.identity(2)
        self.dyn_block[0,1] = self.dt
        self.dyn_block[0:2,-2:] = -np.identity(2)

        self.dyn_vec = np.zeros(2)
        self.dyn_vec[1] = self.g*self.dt

        self.ineq_block = np.zeros((4*self.max_cnts, 2 + self.max_cnts))
        self.ineq_vec = np.zeros(4*self.max_cnts)
        for n in range(self.max_cnts):
            self.ineq_block[4*n, 2 + n] = 1
            self.ineq_block[4*n+1, 2 + n] = -1
            self.ineq_vec[4*n] = self.max_force[n]
            self.ineq_vec[4*n+1] = 0


    def create_constraint_blocks(self, cnt_t):
        '''
        This function creates the constraints matrix for one time step
        Input:
            cnt_t : current contact configuration (1/0 for each contact point)
        '''
        # dynamic constraints
        dyn_block = self.dyn_block.copy()
        dyn_vec = self.dyn_vec.copy()

        for n in range(self.max_cnts):
            dyn_block[1,2+n] = cnt_t[n]*self.dt/self.m
            
        return dyn_block, dyn_vec

    def create_constraints(self, z0, cnt_arr, no_col):
        '''
        This function creates the equality and inequality constraints for the qp
        Input:
            z0: initial position (z, zd)
            cnt_arr : contact array
            r_arr : location of contact points during the plan
            no_cols : number of colocation points
        '''

        A = np.zeros((2*no_col + 2, (2 + self.max_cnts)*no_col + 2))
        b = np.zeros(2*no_col + 2)

        G = np.zeros((4*self.max_cnts*no_col, (2 + self.max_cnts)*no_col + 2))
        h = np.zeros(4*self.max_cnts*no_col)

        for t in range(no_col):
            dyn_block_t, dyn_vec_t = self.create_constraint_blocks(cnt_arr[t])
            A[2*t:2*(t+1), (2+self.max_cnts)*t:(2+self.max_cnts)*(t+1) + 2] = dyn_block_t
            b[2*t:2*(t+1)] = dyn_vec_t

            G[4*self.max_cnts*t:4*self.max_cnts*(t+1), (2+self.max_cnts)*t:(2+self.max_cnts)*(t+1)] = self.ineq_block
            h[4*self.max_cnts*t:4*self.max_cnts*(t+1)] = self.ineq_vec

        A[-2:, 0:2] = np.identity(2)
        b[-2:] = z0

        return A, b, G, h

    def create_cost_matrix(self, zT, w, ter_w, no_col):
        '''
        This function creates the cost matrix for the QP
        Input:
            zT : final desired state [z, zd]
            w : weight matrix (cz, czd, f..)
            ter_w : weight matrix on terminal state
            no_col: number of colocation points
        '''
        
        P = np.zeros(((2 + self.max_cnts)*no_col + 2,(2 + self.max_cnts)*no_col + 2))
        q = np.zeros((2 + self.max_cnts)*no_col + 2)

        wt = np.concatenate((np.tile(w, no_col), ter_w))
        np.fill_diagonal(P, wt)
        
        q[-2:] = -ter_w*zT 

        return P, q

    def optimize(self, z0, cnt_arr, zT, w, ter_w, horizon):
        '''
        This function optimizes the motion
        Input:
            z0 : starting state of the robot
            cnt_arr : contact array
            zT : final desired state
            w : tracking weights
            wT : weight on terminal state
            horizon : duration of motion in seconds
        '''

        no_col = int(np.round(horizon/self.dt, 2))

        assert len(z0) == 2
        assert len(w) == 2 + self.max_cnts
        assert len(ter_w) == 2
        
        A, b, G, h = self.create_constraints(z0, cnt_arr, no_col)
        P, q = self.create_cost_matrix(zT, w, ter_w, no_col)

        qp_G = P
        qp_a = -q
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]    

        sol = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
        
        traj_z = np.zeros((2, no_col + 1))
        fz = np.zeros((self.max_cnts, no_col+1))

        for t in range(no_col+1):
            traj_z[0, t] = sol[t*(2+self.max_cnts)]
            traj_z[1, t] = sol[t*(2+self.max_cnts) + 1]
            if t < no_col:
                fz[:,t] = sol[(2+self.max_cnts)*t + 2:(2+self.max_cnts)*t + 2 + self.max_cnts]
            
        return traj_z, fz

