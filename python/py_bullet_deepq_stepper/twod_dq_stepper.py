import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


class TwoDQStepper:
    def __init__(self, no_actions = [9, 7], lr = 1e-4, gamma = 0.9, use_tarnet = False, trained_model = None):
        '''
        This is a 2d dq stepper.
        State = [x-ux, y - uy, xd, yd, n, action_x, action_y]
        '''
        self.device = torch.device("cpu")
        self.dq_stepper = NN(9, 1).to(self.device) #state+ action -> q_value
        if trained_model:
            self.dq_stepper.load_state_dict(torch.load(trained_model))
            self.dq_stepper.eval()
        self.optimizer = torch.optim.SGD(self.dq_stepper.parameters(), lr)
        self.use_tarnet = use_tarnet
        if self.use_tarnet:
            self.dq_tar_stepper = NN(9, 1).to(self.device)
            self.dq_tar_stepper.load_state_dict(self.dq_stepper.state_dict())
            self.dq_tar_stepper.eval()
        self.gamma = gamma #discount factor
        assert len(no_actions) == 2
        self.no_actions = no_actions
        
        # This is the template of x_in that goes into the dq stepper
        self.x_in = np.zeros((self.no_actions[0]*self.no_actions[1], 9))
        self.x_in[:,7] = np.tile(np.arange(self.no_actions[0]), self.no_actions[1])
        self.x_in[:,8] = np.repeat(np.arange(self.no_actions[1]), self.no_actions[0])
                  
    def predict_action_value(self, x):
        # this function predicts the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6]] = x
        torch_x_in = torch.FloatTensor(self.x_in, device = self.device)
        with torch.no_grad():
            q_values = self.dq_stepper(torch_x_in).detach().numpy()
            action_index = np.argmin(q_values)
            action_x = int(action_index%self.no_actions[0])
            action_y = int(action_index//self.no_actions[0])
        return [action_x, action_y], q_values[action_index]
    
    def tar_predict_action_value(self, x):
        # this function uses tar net to predict 
        # the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6]] = x
        torch_x_in = torch.FloatTensor(self.x_in, device = self.device)
        with torch.no_grad():
            q_values = self.dq_tar_stepper(torch_x_in).detach().numpy()
            action_index = np.argmin(q_values)
            action_x = int(action_index%self.no_actions[0])
            action_y = int(action_index//self.no_actions[0])
        return [action_x, action_y], q_values[action_index]
    
    def predict_eps_greedy(self, x, eps = 0.1):
        # This function returns prediction based on epsillon greedy algorithm
        if np.random.random() > eps:
            return self.predict_action_value(x)[0]
        else:
            action_x = np.random.randint(self.no_actions[0])
            action_y = np.random.randint(self.no_actions[1])
            return [action_x, action_y]
        
    def optimize(self, mini_batch, tau = 0.001):
        # This function performs one step of back propogation for the given mini_batch data
        x_in = torch.FloatTensor(mini_batch[:,0:9].copy(), device = self.device)
        y_train = torch.FloatTensor(mini_batch[:,9].copy(), device = self.device)
        for i in range(len(mini_batch)):
            if not np.isnan(mini_batch[i,10:]).all():
                if not self.use_tarnet:
                    y_train[i] += self.gamma * self.predict_action_value(mini_batch[i,10:])[1]
                else:
                    y_train[i] += self.gamma * self.tar_predict_action_value(mini_batch[i,10:])[1]

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
    
    def predict_q(self, x):
        #for debugging
        # this function predicts the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6]] = x
        torch_x_in = torch.FloatTensor(self.x_in, device = self.device)
        with torch.no_grad():
            q_values = self.dq_stepper(torch_x_in).detach().numpy()
        return q_values

'''
This is an implementation of a 2D Lipm environment with variable number of steps possible
'''

class TwoDLipmEnv:
    
    def __init__(self, h, b, max_step_length, w, no_actions = [11, 9]):
        '''
        Input:
            h : height of the lipm above the ground
            b : width of the base (distance between the feet)
            max_step_length : max step length allowed
            w : weights for the cost computation
            no_actions : number of discretizations
        '''
        
        self.omega = np.sqrt(9.81/h)
        self.max_leg_length = 0.3
        self.dt = 0.001
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
        self.action_space_x = np.around(np.linspace(-max_step_length, max_step_length, no_actions[0]), 2)
        # actions to the free side
        if b > 0 :
            self.action_space_ly = np.geomspace(b, max_step_length + b, int(2*no_actions[1]/3))
            # actions to the non free side where leg can hit the other leg
            # Y axis actions step length allowed such that robot can't step to the left of the left leg
            # or the right to the right leg (no criss crossing)
#             self.action_space_ry = np.arange(0, b, self.action_space_ly[1] - self.action_space_ly[0])
            self.action_space_ry = np.linspace(0, b, int(no_actions[1]/3), endpoint = False)
            self.action_space_y = np.around(np.concatenate((self.action_space_ry, self.action_space_ly)), 2)
        else:
            self.action_space_y = np.around(np.linspace(0, max_step_length, int(no_actions[1]/2)), 2)
            
        self.A = np.matrix([[1, 0, self.dt, 0], 
                            [0, 1, 0, self.dt], 
                            [self.dt*(self.omega**2), 0, 1, 0], 
                            [0, self.dt*(self.omega**2), 0, 1]])
        
        self.B = np.matrix([[0, 0], [0, 0], [-(self.omega**2)*self.dt, 0], [0, -(self.omega**2)*self.dt]])
        self.t = 0
        
    def integrate_lip_dynamics(self, x_t, u_t):
        '''
        integrates the dynamics of the lipm for one time step
        Input:
            x_t : current state of the lip ([x_com, y_com, xd_com, yd_com])
            u_t : current cop location ([u_x, u_y])
        '''
        assert np.shape(x_t) == (4,)
        x_t_1 = np.matmul(self.A, np.transpose(x_t)) + np.matmul(self.B, np.transpose(u_t))
        return x_t_1
    
    def reset_env(self, x0, v_des, epi_time):
        '''
        Resets environment for a new episode
        Input:
            x0 : initial state of the system [x, y, xd, yd]
            v_des : desired velocity [xd_des, yd_des]
            epi_time : episode time
        '''
        assert np.shape(x0) == (4,)
        self.t = 0
        # [x, y, xd, yd, ux, uy, h, n]
        self.sim_data = np.zeros((8, int(epi_time/self.dt)+1))
        self.no_steps = 0
        assert (len(v_des) == 2)
        self.v_des = v_des
        assert (np.linalg.norm([x0[0], self.h]) < self.max_leg_length)
        assert (np.linalg.norm([x0[1], self.h]) < self.max_leg_length)
        self.sim_data[:,0][0:4] = x0
        self.sim_data[:,0][5] = -self.b/2 # right leg on the ground
        self.sim_data[:,0][6] = self.h
        self.sim_data[:,0][7] = 1 # determines which leg is on the ground (1 is right leg)
        
        processed_state = np.zeros(7)
        processed_state[0:5] = np.take(self.sim_data[:,0], [0, 1, 2, 3, 7])
        processed_state[5:7] = self.v_des    
        
        return processed_state
    
    def step_env(self, u, step_time):
        '''
        Integrates the dynamics of the lipm for the duration of a step (until next action is to be taken)
        Input:
            u : action (next step)
            step_time : the duration after which next step is taken
        '''
        for i in range(int(step_time/self.dt)):
            self.sim_data[:,self.t + 1][0:4] = self.integrate_lip_dynamics(self.sim_data[:,self.t][0:4], \
                                                                           self.sim_data[:,self.t][4:6])
            self.sim_data[:,self.t + 1][4:6] = self.sim_data[:,self.t][4:6] # u
            self.sim_data[:,self.t + 1][6] = self.sim_data[:,self.t][6] # h
            self.sim_data[:,self.t + 1][7] = self.sim_data[:,self.t][7] # n
            self.t += 1
        
        self.sim_data[:,self.t][4] += self.action_space_x[u[0]]
        self.sim_data[:,self.t][5] += self.sim_data[:,self.t][7]*self.action_space_y[u[1]]
        self.sim_data[:,self.t][7] = -1*self.sim_data[:,self.t][7]
        
        ## modifying state that is returned is such that the origin is u0 instead of the global origin
        ## This ensures that the state x[0] is bounded by the maximum leg size while collecting data
        processed_state = np.zeros(7)
        processed_state[0:5] = np.take(self.sim_data[:,self.t].copy(), [0, 1, 2, 3, 7]) 
        processed_state[0:2] -= self.sim_data[:,self.t][4:6] # shifting origin to u
        processed_state[5:7] = self.v_des
        
        if not self.isdone():
            self.no_steps += 1
            
        return np.round(processed_state, 2), self.compute_cost(), self.isdone()
    
    def isdone(self):
        '''
        Checks if the kinematic constraints are violated
        '''
        # Computing the hip location
        hip = self.sim_data[:,self.t][0:2].copy()
        hip[1] -= self.sim_data[:,self.t][7]*(self.b/2.0)
        tmp = np.linalg.norm(hip - self.sim_data[:,self.t][4:6])
        current_leg_length = np.linalg.norm([tmp, self.h])
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
        hip = self.sim_data[:,self.t][0:2].copy()
        hip[1] += -1*self.sim_data[:,self.t][7]*(self.b/2) # -1 is to match co ordinate axis
        u = self.sim_data[:,self.t][4:6].copy()
        cost = self.w[0]*(abs(hip - u)[0]) + self.w[0]*(abs(hip - u)[1])
        if self.isdone():
            cost += 100
        cost += self.w[1]*(abs(self.sim_data[:,self.t][2] - self.v_des[0]) \
                               + abs(self.sim_data[:,self.t][3] - self.v_des[1]))
        if np.round(self.sim_data[:,self.t][4] - self.sim_data[:,self.t - 5][4], 2) != 0 or \
            abs(np.round(self.sim_data[:,self.t][5] - self.sim_data[:,self.t - 5][5], 2)) != self.b:
            cost += self.w[2]

        return cost
    
    def random_action(self):
        '''
        Genarates random action
        '''
        action_x = np.random.randint(len(self.action_space_x))
        action_y = np.random.randint(len(self.action_space_y))
        
        return np.array([action_x, action_y])

    