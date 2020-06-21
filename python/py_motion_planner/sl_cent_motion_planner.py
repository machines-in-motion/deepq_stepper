## This file containts the optimizer to for the spring loaded centeroidal
## dynamics 
## Author : Avadesh Meduri
## Date : 17/06/2020

from matplotlib import pyplot as plt
import numpy as np 
from quadprog import solve_qp

class SLCentMotionPlanner:

    def __init__(self, dt, max_cnts, mass, inertia, max_force, max_ht):
        '''
        Input:
            dt: time discretization
            max_cnts : maximum number of contact points
            mass : mass of the robot
            inertia : inertia
            max_force : maximum force at each contact point
            max_ht : maximum z difference between com and contact point
        '''
    
        self.dt = dt
        self.max_cnts = max_cnts
        self.m = mass
        self.g = 9.81
        # This is a problem as its not fixed (find alternative)
        self.inertia = inertia
        assert len(max_force) == self.max_cnts
        assert len(max_ht) == self.max_cnts
        self.max_force = max_force
        self.max_ht = max_ht

    def xy_dynamics(self, K, xy, x_ori, r, cnt_array):
        '''
        This function returns dy/dt = f(y) for the xy dynamics
        Input:
            xy : center of mass location in xy plane and yaw, velocity in xy and yaw vel
            x_ori : orientation of the base (roll, pitch, yaw) #assumed to be constant
            r : contact location
            cnt_array : contact array (1 if in contact, 0 otherwise)
        '''
        d_xy = np.zeros(6)
        d_xy[0:3] = xy[3:]
        for n in range(self.max_cnts):    
            d_xy[3] += cnt_array[n]*(K[n][0]*(xy[0] - r[n][0]))/self.m
            d_xy[4] += cnt_array[n]*(K[n][1]*(xy[1] - r[n][1]))/self.m
            d_xy[5] += cnt_array[n]*(K[n][0]*(r[n][1] - xy[1])*(r[n][0] - xy[0]) + K[n][1]*(xy[0] - r[n][0])*(r[n][1] - xy[1]))/self.inertia

        return d_xy
    
    def rk_integrate(self, K, xy, x_ori, r, cnt_array):
        '''
        Integrates dynamics using runga kutta 4th order method
        Input:
            x : center of mass location in xy plane and yaw, velocity in xy and yaw vel
            x_ori : orientation of the base (roll, pitch, yaw) #assumed to be constant
            r : contact location
            cnt_array : contact array (1 if in contact, 0 otherwise)
        '''

        k1 = self.xy_dynamics(K, xy, x_ori, r, cnt_array)
        k2 = self.xy_dynamics(K, xy + 0.5*self.dt*k1, x_ori, r, cnt_array)
        k3 = self.xy_dynamics(K, xy + 0.5*self.dt*k2, x_ori, r, cnt_array)
        k4 = self.xy_dynamics(K, xy + self.dt*k3, x_ori, r, cnt_array)
        
        return np.add(xy, self.dt*(k1 + 2*k2 + 2*k3 + k4)/6)

    def integrate_xy_dynamics(self, K_arr, xy0, x_ori, cnt_plan, horizon):
        '''
        Integrates xy dynamics in the duration of horizon
        Input:
            K_arr : spring stiffness array
            xy0 : starting config in xy plane (x,y,yaw, xd, yd, ang_vel_yaw)
            x_ori : orientation of the base (roll, pitch, yaw) #assumed to be constant
            cnt_plan : contact plan
            horizon : horizon
        '''

        cnt_array, r_arr = self.create_contact_array(cnt_plan, horizon)

        self.xy_sim_data = np.zeros((6, int(horizon/self.dt)))
        self.xy_sim_data[:,0] = xy0
        for t in range(0, int(horizon/self.dt)-1):
            self.xy_sim_data[:,t + 1] = self.rk_integrate(K_arr[t], self.xy_sim_data[:,t], x_ori, r_arr[t], cnt_array[t])

        return self.xy_sim_data

    def create_dynamic_constraints(self, K, xy_t, r, cnt_array):
        '''
        This function creates the dynamics constraints block for a given time step
        Input:
            K : spring constant at given time step
            xy_t : state in xy plane (x, y, yaw, xd, yd, ang_vel_yaw)
            h_t : hip location in xy plane
            r : contact locations at time step
            cnt_array : contact configuration at given time step
        '''
        const_matrix = np.zeros((6, 6 + self.max_cnts))
        const_vec = np.zeros(6)
        
        const_matrix[0:6,0:6] = np.identity(6)
        const_matrix[0,1] = self.dt #cz
        const_vec[1] = self.g*self.dt
        const_matrix[2,4] = self.dt #roll
        const_matrix[3,5] = self.dt #pitch
        for n in range(self.max_cnts):
            const_matrix[1, 6+n] = cnt_array[n]*self.dt/self.m # dcz
            const_matrix[4,0] += -cnt_array[n]*K[n][1]*(r[n][1] - xy_t[1])*self.dt/self.inertia
            const_matrix[4,6+n] = cnt_array[n]*(r[n][1] - xy_t[1])*self.dt/self.inertia
            const_vec[4] += -cnt_array[n]*K[n][1]*(r[n][2])*(r[n][1] - xy_t[1])*self.dt/self.inertia

            const_matrix[5,0] += cnt_array[n]*K[n][0]*(r[n][0] - xy_t[0])*self.dt /self.inertia
            const_matrix[5,6+n] = cnt_array[n]*(xy_t[0] - r[n][0])*self.dt/self.inertia
            const_vec[5] += cnt_array[n]*K[n][0]*(r[n][2])*(r[n][0] - xy_t[0])*self.dt/self.inertia

        return const_matrix, const_vec

    def create_inequality_constraints(self, r, cnt_array):
        '''
        This function creates the inequality constraints for torque and kinematic limits
        '''
        ineq_matrix = np.zeros(((2 + 2)*self.max_cnts, 6 + self.max_cnts))
        ineq_vec = np.zeros((2+2)*self.max_cnts)

        for n in range(self.max_cnts):
            ineq_matrix[4*n,6+n] = -1*cnt_array[n]
            ineq_matrix[4*n+1,6+n] = 1*cnt_array[n]
            ineq_vec[4*n+1] = self.max_force[n]*cnt_array[n]
            
            ineq_matrix[4*n+2,0] = 1*cnt_array[n]
            ineq_vec[4*n+2] = cnt_array[n]*(r[n][2] + self.max_ht[n])
            ineq_matrix[4*n+3,0] = -1*cnt_array[n]
            ineq_vec[4*n+3] = -cnt_array[n]*(r[n][2])

        return ineq_matrix, ineq_vec

    def create_contact_array(self, cnt_plan, horizon):
        '''
        This function converts contact plan into an array to create constraints
        Input:
            cnt_plan : [[[1/0, location of contact(x,y,z), start time, end time],[]], ...]
            horizon : total duration of plan
        '''
        assert np.shape(cnt_plan)[1] == self.max_cnts
        assert np.shape(cnt_plan)[2] == 6

        cnt_array = np.zeros((int(horizon/self.dt), self.max_cnts))
        r_arr = np.zeros((int(horizon/self.dt), self.max_cnts, 3))
        t_arr = np.zeros(self.max_cnts)
        for i in range(len(cnt_plan)):
            for n in range(self.max_cnts):
                duration = np.round((cnt_plan[i][n][5] - cnt_plan[i][n][4])/self.dt, 1)
                cnt_array[int(t_arr[n]):int(t_arr[n]+duration), n] = cnt_plan[i][n][0]
                r_arr[int(t_arr[n]):int(t_arr[n]+duration), n] = cnt_plan[i][n][1:4]
                t_arr[n] += duration

        return cnt_array, r_arr

    def create_constraints(self, K_arr, x0, r_arr, cnt_array, horizon, xT):
        '''
        This function creates the equality and inequality constraint matrices
        to generate fz
        Input:
            K_arr : the array of spring stiffness at each time step
            x0; initial state (x,y,z, xd, yd, zd, roll, pitch, yaw, ang vel)
            r_arr : constact location array
            cnt_array : contact configuration array
            horizon : duration to optimize
            xT : final desired state at the end of the motion
        '''

        no_col_points = int(horizon/self.dt) # number of colocation points
        # the last 6 variables are slack variables for terminal state
        A = np.zeros((6*no_col_points + 2*6, (6+self.max_cnts)*(no_col_points) + 2*6))
        b = np.zeros(6*no_col_points + 2*6)

        G = np.zeros((4*self.max_cnts*no_col_points, (6+self.max_cnts)*(no_col_points) + 2*6))
        h = np.zeros(4*self.max_cnts*no_col_points)

        self.xy_sim_data = np.zeros((6, no_col_points+1))
        self.xy_sim_data[:,0] = np.take(x0, [0, 1, 8, 3, 4, 11])
        for t in range(0, no_col_points):
            self.xy_sim_data[:,t + 1] = self.rk_integrate(K_arr[t], self.xy_sim_data[:,t], x0[3:6], r_arr[t], cnt_array[t])
            A[6*t:6*(t+1), (6 + self.max_cnts)*t: (6 + self.max_cnts)*(t+1)], b[6*t:6*(t+1)] = \
                self.create_dynamic_constraints(K_arr[t], self.xy_sim_data[:,t], r_arr[t], cnt_array[t])
            A[6*t:6*(t+1), (6 + self.max_cnts)*(t+1):(6 + self.max_cnts)*(t+1) + 6] = -np.identity(6)     
            
            # # inequality constraints
            G[4*self.max_cnts*t:4*self.max_cnts*(t+1), (6 + self.max_cnts)*t: (6 + self.max_cnts)*(t+1)], \
                h[4*self.max_cnts*t:4*self.max_cnts*(t+1)] = self.create_inequality_constraints(r_arr[t], cnt_array[t])

        # initial value constraints
        A[-12:-6, 0:6] = np.identity(6)
        b[-12:-6] = np.take(x0,(2,5,6,7,9,10))
        # terminal slack variables
        A[-6:, -12:-6] = np.identity(6)
        A[-6:, -6:] = np.identity(6)
        b[-6:] = np.take(xT, (2, 5, 6, 7, 9, 10))
        return A, b, G, h

    def create_cost_matrix(self, horizon, w):
        '''
        This function creates the cost matrix
        Input: 
            horizon : duration of optimization
        '''
        no_col_points = int(horizon/self.dt) # number of colocation points
        wt = np.zeros(6+self.max_cnts)
        wt[0:6] = [1e-4,1e-4,w[1],w[1],w[2],w[2]]
        wt[6:] = w[0]
        P = np.zeros(((6+self.max_cnts)*(no_col_points) + 2*6, (6+self.max_cnts)*(no_col_points) + 2*6))
        np.fill_diagonal(P, np.tile(wt, (6+self.max_cnts)*no_col_points + 2*6))
        P[-1,-1], P[-2,-2], P[-3,-3], P[-4,-4], P[-5,-5], P[-6,-6] = 6*[w[3]]

        q = np.zeros((6+self.max_cnts)*(no_col_points) + 2*6)

        return P, q


    def optimize_motion(self, cnt_plan, K_arr, x0, horizon, xT, w):
        '''
        This function optimizes the motion
        Input:
            cnt_plan : cnt plan
            K_arr : spring stiffness for each leg at all time steps
            x0: starting configuration of dynamics
            horizon : duration of motion
            xT : final desired state
            w : weight matrix
                [cost on fz, cost on angular position, angular vel, terminal state]
        '''

        cnt_array, r_arr = self.create_contact_array(cnt_plan, horizon)
        P, q = self.create_cost_matrix(horizon, w)
        A, b, G, h = self.create_constraints(K_arr, x0, r_arr, cnt_array, horizon, xT)

        qp_G = P
        qp_a = -q
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]    

        sol = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
        traj = np.zeros((12, int(horizon/self.dt)+1))
        u = np.zeros((self.max_cnts, int(horizon/self.dt)+1))
        traj[0:2] = self.xy_sim_data[0:2]
        traj[2] = sol[0:-6:6+self.max_cnts]
        traj[3:5] = self.xy_sim_data[3:5]
        traj[5] = sol[1:-6:6+self.max_cnts]
        traj[6] = sol[2:-6:6+self.max_cnts]
        traj[7] = sol[3:-6:6+self.max_cnts]
        traj[8] = self.xy_sim_data[2]
        traj[9] = sol[4:-6:6+self.max_cnts]
        traj[10] = sol[5:-6:6+self.max_cnts]
        traj[11] = self.xy_sim_data[5]
        for n in range(self.max_cnts):
            u[n,:-1] = sol[6+n:-6:6+self.max_cnts]
            u[n,-1] = u[n,-2]

        self.traj = traj
        self.u = u

        return traj, u

    def plot_traj(self, horizon):
        
        t = np.linspace(0, horizon, int(horizon/self.dt) + 1)

        fig, axs = plt.subplots(4,1)
        axs[0].plot(t, self.traj[0], label = 'com_x')
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(t, self.traj[1], label = 'com_y')
        axs[1].grid()
        axs[1].legend()
        axs[2].plot(t, self.traj[2], label = 'com_z')
        axs[2].grid()
        axs[2].legend()
        for n in range(self.max_cnts):
            axs[3].plot(t, self.u[n], label = 'f' + str(n))
        axs[3].grid()
        axs[3].legend()
        
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t, self.traj[3], label = 'com_xd')
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(t, self.traj[4], label = 'com_yd')
        axs[1].grid()
        axs[1].legend()
        axs[2].plot(t, self.traj[5], label = 'com_zd')
        axs[2].grid()
        axs[2].legend()
        
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t, self.traj[6], label = 'com_ang_x')
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(t, self.traj[7], label = 'com_ang_y')
        axs[1].grid()
        axs[1].legend()
        axs[2].plot(t, self.traj[8], label = 'com_ang_z')
        axs[2].grid()
        axs[2].legend()
        
        plt.show()



# dt = 0.05
# hor = 0.8
# k = [[9.81/0.2, 0.0], [9.81/0.2, 0.0]]
# max_force = [20,20]
# max_ht = [0.25, 0.25]

# cnt_plan = [[[0, 0.0, 0.13, 0, 0, 0.2],[1, 0.1, -0.13, 0, 0, 0.2]],
#             [[1, 0.2, 0.13, 0, 0.2, 0.4],[0, 0.0, -.13, 0, 0.2, 0.4]],
#             [[0, 0.0, 0.13, 0, 0.4, 0.6],[1, 0.3, -0.13, 0, 0.4, 0.6]],
#             [[1, 0.4, 0.13, 0, 0.6, 0.8],[0, 0.0, -.13, 0, 0.6, 0.8]]]

# K_arr = np.tile(k,(int(hor/dt), 1, 1))
# x0 = [0, 0, 0.2, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]
# xT = [0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# w = [1e+2, 1, 1, 1e+7]

# mp = SLCentMotionPlanner(dt, 2, 1, 1, max_force, max_ht)
# motion, forces = mp.optimize_motion(cnt_plan, K_arr, x0, hor, xT, w)
# mp.plot_traj(hor)