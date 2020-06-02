## This file generates the force profile for the inverted pendulum model
## Author : Avadesh Meduri
## Date : 1/06/2020

from matplotlib import pyplot as plt
import numpy as np 
from quadprog import solve_qp

class IPMotionPlanner:

    def __init__(self, dt, max_torque):
        
        self.dt = dt
        self.max_torque = max_torque
        self.dyn_block = np.matrix([[1, self.dt, 0, -1, 0],
                                    [0, 1, self.dt, 0, -1]])
        self.torque_block = np.matrix([[0, 0, 1],
                                       [0, 0, -1]])

    def create_cost_matrix(self, n):
        '''
        This function forms the weight matrix for the qp
        3n-1 decision variables plus one slack variable
        Input:
            n : number of discrete steps
        '''
        P = np.zeros((3*n, 3*n))
        np.fill_diagonal(P, np.tile([1e-3, 1e-3, 1.0], n))
        P[3*n-1][3*n-1] = 1e+10
        q = np.zeros(3*n)

        return P, q

    def create_constraints(self, n, max_torque, start, step_size):
        '''
        This function creates the equality constraints in the qp corresponding to the
        dynamics and inequality constraints for torque limits
        Input:
            n: number of discrete steps
            max_torques : torque limits
        '''
        A = np.zeros((2*(n-1) + 4, 3*n))
        b = np.zeros(2*(n-1) + 4)
        G = np.zeros((2*(n-1), 3*n))
        h = max_torque*np.ones(2*(n-1))
        
        for t in range(0, n-1):
            A[2*t:2*(t+1), 3*t:(3*t)+5] = self.dyn_block
            G[2*t:2*(t+1), 3*t:3*t+3] = self.torque_block

        A[-4][0] = 1
        A[-3][1] = 1
        A[-2][-3] = 1
        A[-2][-1] = 1
        A[-1][-2] = 1

        b[-4] = start
        b[-2] = start + step_size

        return G, h, A, b

    def generate_force_trajectory(self, start, step_size, step_time):
        '''
        This function generates the force profile to move the inverted pendulum
        to the desired height in the z direction
        Input:
            start : current height of the center of mass
            step_size : height of the step
            step_time : time after which step is taken
        '''

        n = int(step_time /self.dt)

        P, q = self.create_cost_matrix(n)
        G, h, A, b = self.create_constraints(n, self.max_torque, start, step_size)

        qp_G = P
        qp_a = -q
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]    

        sol = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

        x = sol[0:-2:3]
        xd = sol[1:-2:3]
        u = sol[2:-2:3]

        return x, xd, u

ip = IPMotionPlanner(0.01, 5)
x, xd, u = ip.generate_force_trajectory(0.2, -0.05, 0.1) 

# plt.plot(x)
# plt.show()
# plt.plot(u)
# plt.show()