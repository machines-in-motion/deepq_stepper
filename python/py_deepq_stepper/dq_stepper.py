import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from py_deepq_stepper.motion_planner import IPMotionPlanner

class NN(nn.Module):
    def __init__(self, inp_size, out_size):
        
        super(NN, self).__init__()
        self.l1 = nn.Linear(inp_size, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l8 = nn.Linear(512, out_size)
    
    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = self.l8(x)
        return x


class DQStepper:
    def __init__(self, no_actions = [9, 7], lr = 1e-4, gamma = 0.9, use_tarnet = False, trained_model = None):
        '''
        This is a 3d dq stepper.
        State = [x-ux, y-uy, z-uz, xd, yd, n, action_x, action_y, action_z]
        '''
        self.device = torch.device("cpu")
        self.dq_stepper = NN(11, 1).to(self.device) #state+ action -> q_value
        if trained_model:
            self.dq_stepper.load_state_dict(torch.load(trained_model))
            self.dq_stepper.eval()
        self.optimizer = torch.optim.SGD(self.dq_stepper.parameters(), lr)
        self.use_tarnet = use_tarnet
        if self.use_tarnet:
            self.dq_tar_stepper = NN(11, 1).to(self.device)
            self.dq_tar_stepper.load_state_dict(self.dq_stepper.state_dict())
            self.dq_tar_stepper.eval()
        self.gamma = gamma #discount factor
        assert len(no_actions) == 2
        self.no_actions = no_actions
        
        # This is the template of x_in that goes into the dq stepper
        self.max_step_height = 0.04
        self.delta = 4 #discretization of step height
        self.x_in = np.zeros((self.no_actions[0]*self.no_actions[1], 11))
        self.x_in[:,8] = np.tile(np.arange(self.no_actions[0]), self.no_actions[1])
        self.x_in[:,9] = np.repeat(np.arange(self.no_actions[1]), self.no_actions[0])
                  
    def predict_action_value(self, x):
        # this function predicts the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = x
        self.x_in[:,10] = (1/(self.delta))*self.max_step_height*\
                            np.random.randint(-self.delta, self.delta+1, self.no_actions[0]*self.no_actions[1])
        torch_x_in = torch.FloatTensor(self.x_in, device = self.device)
        with torch.no_grad():
            q_values = self.dq_stepper(torch_x_in).detach().numpy()
            action_index = np.argmin(q_values)
            action_x = int(action_index%self.no_actions[0])
            action_y = int(action_index//self.no_actions[0])
            action_z = self.x_in[action_index,10]
        return [action_x, action_y, action_z], q_values[action_index]
    
    def tar_predict_action_value(self, x):
        # this function uses tar net to predict 
        # the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = x
        self.x_in[:,10] = (1/(self.delta))*self.max_step_height*\
                            np.random.randint(-self.delta, self.delta+1, self.no_actions[0]*self.no_actions[1])
        torch_x_in = torch.FloatTensor(self.x_in, device = self.device)
        with torch.no_grad():
            q_values = self.dq_tar_stepper(torch_x_in).detach().numpy()
            action_index = np.argmin(q_values)
            action_x = int(action_index%self.no_actions[0])
            action_y = int(action_index//self.no_actions[0])
            action_z = self.x_in[action_index,10]
        return [action_x, action_y, action_z], q_values[action_index]
    
    def predict_eps_greedy(self, x, eps = 0.1):
        # This function returns prediction based on epsillon greedy algorithm
        if np.random.random() > eps:
            return self.predict_action_value(x)[0]
        else:
            action_x = np.random.randint(self.no_actions[0])
            action_y = np.random.randint(self.no_actions[1])
            action_z = 2*self.max_step_height*(np.random.rand() - 0.5)
            return [action_x, action_y, action_z]
        
    def optimize(self, mini_batch, tau = 0.001):
        # This function performs one step of back propogation for the given mini_batch data
        x_in = torch.FloatTensor(mini_batch[:,0:11].copy(), device = self.device)
        y_train = torch.FloatTensor(mini_batch[:,11].copy(), device = self.device)
        for i in range(len(mini_batch)):
            if not np.isnan(mini_batch[i,12:]).all():
                if not self.use_tarnet:
                    y_train[i] += self.gamma * self.predict_action_value(mini_batch[i,12:])[1]
                else:
                    y_train[i] += self.gamma * self.tar_predict_action_value(mini_batch[i,12:])[1]

        y_train = y_train.unsqueeze(1).detach() #ensures that gradients are not computed on this
        x_train = self.dq_stepper(x_in)

        loss = F.mse_loss(x_train, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.use_tarnet:
            for tar_param, param in zip(self.dq_tar_stepper.parameters(), self.dq_stepper.parameters()):
                tar_param.data.copy_(tar_param.data * (1.0 - tau) + param.data * tau)
                
        return loss
        
    def predict_q(self, x, terrain):
        #for debugging
        # this function predicts the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = x
        self.x_in[:,10] = terrain
        torch_x_in = torch.FloatTensor(self.x_in, device = self.device)
        with torch.no_grad():
            q_values = self.dq_stepper(torch_x_in).detach().numpy()
            action_index = np.argmin(q_values)
            action_x = int(action_index%self.no_actions[0])
            action_y = int(action_index//self.no_actions[0])
            action_z = self.x_in[action_index,10]
            
        return q_values, [action_x, action_y, action_z] 

class InvertedPendulumEnv:
    
    def __init__(self, h, b, max_step_length, w, no_actions = [11, 9]):
        '''
        Input:
            h : height of the lipm above the ground at the start of the step
            b : width of the base (distance between the feet)
            max_step_length : max step length allowed
            w : weights for the cost computation
            no_actions : number of discretizations
        '''
        self.g = 9.81
        self.max_leg_length = 0.32
        # maximum accelertion in the z direction by applying force on the ground
        self.max_acc = 7.0
        self.dt = 0.001
        # nominal desired hight of pendulum above the ground (need not be satisfied at all times)
        # is a soft constraint in the qp
        self.h = h
        self.b = b
        self.no_steps = 0
        assert len(w) == 3
        self.w = w
        assert (np.linalg.norm([max_step_length, self.h]) < self.max_leg_length)
        assert len(no_actions) == 2
        # The co ordinate axis is x : forward and y : sideways walking, z : faces upward
        # This means that left leg is on the positive side of the y axis
        # The addition b is added to accomodate a step length larger than leg length as it may be feasible
        # in high velocity cases.
        self.action_space_x = np.around(np.linspace(-1.2*max_step_length, 1.2*max_step_length, no_actions[0]), 2)
        # actions to the free side
        if b > 0 :
            self.action_space_ly = np.geomspace(b, 1.5*max_step_length + b, int(2*no_actions[1]/3))
            # actions to the non free side where leg can hit the other leg
            # Y axis actions step length allowed such that robot can't step to the left of the left leg
            # or the right to the right leg (no criss crossing)
            self.action_space_ry = np.linspace(0, b, int(no_actions[1]/3), endpoint = False)
            self.action_space_y = np.around(np.concatenate((self.action_space_ry, self.action_space_ly)), 2)
        else:
            self.action_space_y = np.around(np.linspace(0, max_step_length, int(no_actions[1]/2)), 2)
        
        self.t = 0
        self.max_step_height = 0.05
        # QP parameters
        self.delta_t = 0.01
        self.ipmotionplanner = IPMotionPlanner(self.delta_t, self.max_acc)
        
        
    def integrate_ip_dynamics(self, x_t, u_t, z_acc):
        '''
        This function integrated the ip dynamics for one step using euler integration scheme
        Input:
            x_t : state at time step t (cx, cy, cz, cxd, cyd, czd)
            u_t : Cop location at time step t (ux, uy, uz)
            z_acc : acceleration in z direction (control input to increase height)
        '''
        
        x_t_1 = np.zeros(6)
        x_t_1[0:3] = np.add(x_t[0:3], x_t[3:]*self.dt)
        x_t_1[3] = x_t[3] + ((z_acc + self.g)*(x_t[0] - u_t[0])/(x_t[2] - u_t[2]))*self.dt
        x_t_1[4] = x_t[4] + ((z_acc + self.g)*(x_t[1] - u_t[1])/(x_t[2] - u_t[2]))*self.dt
        x_t_1[5] = x_t[5] + z_acc*self.dt

        return x_t_1
    
    def reset_env(self, x0, v_des, epi_time):
        '''
        Resets environment for a new episode
        Input:
            x0 : initial state of the system [x, y, z, xd, yd, zd]
            v_des : desired velocity [xd_des, yd_des]
            epi_time : episode time
        '''
        assert len(x0) == 5
        self.t = 0
        # [x, y, z, xd, yd, zd, ux, uy, uz, n]
        self.sim_data = np.zeros((10, int(epi_time/self.dt)+1))
        self.no_steps = 0
        assert (len(v_des) == 2)
        self.v_des = v_des
        assert (np.linalg.norm([x0[0], x0[2]]) < self.max_leg_length)
        assert (np.linalg.norm([x0[1], x0[2]]) < self.max_leg_length)
        self.sim_data[:,0][0:5] = x0
        self.sim_data[:,0][7] = -self.b/2 # right leg on the ground
        self.sim_data[:,0][9] = 1 # determines which leg is on the ground (1 is right leg)
        
        processed_state = np.zeros(8)
        processed_state[0:6] = np.take(self.sim_data[:,0], [0, 1, 2, 3, 4, 9])
        processed_state[6:8] = self.v_des    
        
        return processed_state
    
    
    def step_env(self, u, step_time):
        '''
        This function simulates the environment for one foot step
        Input:
            u : next step location
            step_time : duration of after which step is taken [ux_index, uy_index, uz (value)]
        '''
    
        assert u[2] < 0.07
        x , xd, acc = self.ipmotionplanner.generate_force_trajectory(self.sim_data[:,self.t][2], \
                                                             self.sim_data[:,self.t][8] + u[2], step_time, self.h)
        acc = np.repeat(acc, int(self.delta_t/self.dt))
        for i in range(int(step_time/self.dt)-1):
            self.sim_data[:,self.t+1][0:6] = self.integrate_ip_dynamics(self.sim_data[:,self.t][0:6], \
                                                                   self.sim_data[:,self.t][6:9], acc[i]) 
            self.sim_data[:,self.t+1][6:9] = self.sim_data[:,self.t][6:9] #u
            self.sim_data[:,self.t+1][9] = self.sim_data[:,self.t][9] #n
            self.t += 1

        self.sim_data[:,self.t][6] += self.action_space_x[u[0]]
        self.sim_data[:,self.t][7] += self.sim_data[:,self.t][9]*self.action_space_y[u[1]]
        self.sim_data[:,self.t][8] += u[2]
        self.sim_data[:,self.t][9] = -1*self.sim_data[:,self.t][9]
        
        ## modifying state that is returned is such that the origin is u0 instead of the global origin
        ## This ensures that the state x[0] is bounded by the maximum leg size while collecting data
        processed_state = np.zeros(8)
        processed_state[0:6] = np.take(self.sim_data[:,self.t], [0, 1, 2, 3, 4, 9])
        processed_state[0:3] -= self.sim_data[:,self.t][6:9] # shifting origin to u
        processed_state[6:8] = self.v_des    
        
        if not self.isdone():
            self.no_steps += 1
        
        return np.round(processed_state, 2), self.compute_cost(), self.isdone()
    
    def isdone(self):
        '''
        Checks if the kinematic constraints are violated
        '''
        # Computing the hip location
        hip = self.sim_data[:,self.t][0:2].copy()
        hip[1] -= self.sim_data[:,self.t][9]*(self.b/2.0)
        tmp = np.linalg.norm(hip - self.sim_data[:,self.t][6:8])
        h = self.sim_data[:,self.t][2] - self.sim_data[:,self.t][8]
        current_leg_length = np.linalg.norm([tmp, h])
        if current_leg_length > self.max_leg_length:
            return True
        else:
            return False
        
    def compute_cost(self):
        '''
        Computes cost which is distance between the hip(closest hip depending on which foot is on the ground)
        and the foot + velocity of the center of mass + 1 if step length not equal to zero (after taking into
        account the offset) + 100 if episode terminates (kinematics constraints are violated)
        '''
        hip = self.sim_data[:,self.t][0:3].copy()
        hip[1] += -1*self.sim_data[:,self.t][9]*(self.b/2) # -1 is to match co ordinate axis
        u = self.sim_data[:,self.t][6:9].copy()
        cost = self.w[0]*(abs(hip - u)[0]) + self.w[0]*(abs(hip - u)[1])
        if self.isdone():
            cost += 100
        cost += self.w[1]*(abs(self.sim_data[:,self.t][3] - self.v_des[0]) \
                               + abs(self.sim_data[:,self.t][4] - self.v_des[1]))
        
        cost += self.w[2]*(abs(np.round(self.sim_data[:,self.t][6] - self.sim_data[:,self.t - 5][6], 2)))
        cost += abs(self.w[2]*(abs(np.round(self.sim_data[:,self.t][7] - self.sim_data[:,self.t - 5][7], 2)) - self.b))
        cost += self.w[2]*(abs(np.round(self.sim_data[:,self.t][8] - self.sim_data[:,self.t - 5][8], 2)))
    
        return cost
    
    def random_action(self):
        '''
        Genarates random action
        '''
        action_x = np.random.randint(len(self.action_space_x))
        action_y = np.random.randint(len(self.action_space_y))
        action_z = self.max_step_height*(np.random.rand() - 0.5)
        
        return [action_x, action_y, action_z]

    