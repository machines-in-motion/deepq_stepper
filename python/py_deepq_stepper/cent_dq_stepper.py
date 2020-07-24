import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from py_motion_planner.cent_motion_planner import CentMotionPlanner

# Lipm - 8 layers , 512 each
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
        self.l8 = nn.Linear(512, 512)
        self.l9 = nn.Linear(512, out_size)
    
    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        x = self.l9(x)
        return x

class DQStepper:
    def __init__(self, env, lr = 1e-4, gamma = 0.9, use_tarnet = False, trained_model = None):
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
        self.no_actions = env.no_actions
        
        # This is the template of x_in that goes into the dq stepper
        self.max_step_height = 0.00
        self.max_no = 5 #number of actions with non zero step in z
        self.x_in = np.zeros((self.no_actions[0]*self.no_actions[1], 11))
        self.x_in[:,8] = np.tile(np.arange(self.no_actions[0]), self.no_actions[1])
        self.x_in[:,9] = np.repeat(np.arange(self.no_actions[1]), self.no_actions[0])
        
    def predict_action_value(self, x):
        # this function predicts the q_value for different actions and returns action and min q value
        self.x_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = x
        for e in np.random.randint(0, len(self.x_in), self.max_no):
            self.x_in[e, 10] = 2*self.max_step_height*(np.random.rand() - 0.5)
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
        for e in np.random.randint(0, len(self.x_in), self.max_no):
            self.x_in[e, 10] = 2*self.max_step_height*(np.random.rand() - 0.5)
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

'''
This is an implementation of an environment with Centroidal dynamics to train
dq steper for running. CentMotionPlanner generates trajectories with zero momentum
'''

class CentEnv:
    
    def __init__(self, h, b, max_step_length, w_cost, no_actions=[11,9]):
        '''
        Input:
            h : height of the com above the ground at the start of the step
            b : width of the base (distance between the feet)
            k : spting stiffness for each leg
            max_step_length : max step length allowed
            w_cost : weights for the cost computation
            no_actions : number of discretizations
        '''
        self.g = 9.81
        self.max_leg_length = 0.32
        self.mass = 1.27
        self.inertia = [0.016, 0.031, 0.031]
        self.com_offset = 0.078
        # nominal desired height above the foot location
        self.h = h
        self.b = b
        self.no_steps = 0
        self.max_step_ht = 0.1 # maximum step in z
        self.no_actions = no_actions
        assert len(w_cost) == 3
        self.w_cost = w_cost #wt in cost computations
        assert len(no_actions) == 2
        # The co ordinate axis is x : forward and y : sideways walking, z : faces upward
        # This means that left leg is on the positive side of the y axis
        # The addition b is added to accomodate a step length larger than leg length as it may be feasible
        # in high velocity cases.
        
        if no_actions[0] == 1:
            self.action_space_x = [0.0]
        else:
            self.action_space_x = np.around(np.linspace(-max_step_length, max_step_length, no_actions[0]), 2)
        
        # actions to the free side
        if b > 0 :
            self.action_space_ly = np.geomspace(b, max_step_length/1.0 + b, int(6*no_actions[1]/9))
            # actions to the non free side where leg can hit the other leg
            # Y axis actions step length allowed such that robot can't step to the left of the left leg
            # or the right to the right leg (no criss crossing)
            self.action_space_ry = np.linspace(0, b, int(3*no_actions[1]/9), endpoint = False)
            self.action_space_y = np.around(np.concatenate((self.action_space_ry, self.action_space_ly)), 2)
        
        else:
            self.action_space_y = np.around(np.linspace(0, max_step_length/1.0, int(no_actions[1])), 2)
        
        self.t = 0
        # motion planner params
        self.delta_t = 0.025
        self.f_max = np.array([[30,30, 30], [30, 30, 30]])
        self.max_ht = np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]])
        self.w = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e+2, 1e+2, 1e+7, 1e+7, 1e-4, 1e-4, 1e+1, 1e-4, 1e-4, 1e+1])
        self.ter_w = np.array([1e-4, 1e-4, 1e+8, 1e-4, 1e-4, 1e+5, 1e+3, 1e+3, 1e+6, 1e+6])
        self.xt = [0, 0, self.h, 0, 0, 0, 0, 0, 0, 0]
        self.cent_mp = CentMotionPlanner(self.delta_t, 2, self.mass, self.inertia, self.f_max, self.max_ht)
        
    def reset_env(self, x0, v_des, epi_time):
        '''
        This function resets the environment
        Input:
            x0 : starting state [x,y,z,xd,yd,zd]
            v_des : desired velocity of center of mass
            epi_time : episode time
        '''
        assert len(x0) == 6
        self.t = 0
        self.v_des = v_des
        self.sim_data = np.zeros((10 + 3 + 1, int(np.round(epi_time/self.delta_t,2))+1))
        self.sim_data[:,self.t][0:6] = x0
        self.sim_data[:,self.t][11] = -self.b/2
        self.sim_data[:,self.t][13] = 1 # right leg on the ground
        
        processed_state = np.zeros(8)
        processed_state[0:6] = np.take(self.sim_data[:,self.t], [0, 1, 2, 3, 4, 13])
        processed_state[0:3] -= self.sim_data[:,self.t][10:13]
        processed_state[6:] = self.v_des    
        
        return processed_state
    
    def step_env(self, action, step_time, air_time):
        '''
        This function simulates the environment for one foot step
        Input:
            u : next step location
            step_time : duration of after which step is taken [ux_index, uy_index, uz (value)]
        '''
        assert len(action) == 3
        assert action[2] < self.max_step_ht

        self.xt[2] = self.sim_data[12, self.t] + action[2] + self.h
        horizon = 2*step_time + air_time
        cnt_plan = [[[0, 0, 0, 0, 0, step_time], [0, 0, 0, 0, 0, step_time]],
                    [[0, 0, 0, 0, step_time, step_time + air_time], [0, 0, 0, 0, step_time, step_time + air_time]],
                    [[0, 0, 0, 0, step_time + air_time, np.round(2*step_time + air_time,2)], [0, 0, 0, 0, step_time + air_time, np.round(2*step_time + air_time,2)]]]

        
        if self.sim_data[:,self.t][13] > 0:
            cnt_plan[0][1][0] = 1
            cnt_plan[0][1][1:4] = self.sim_data[:,self.t][10:13]
            cnt_plan[2][0][0] = 1
            cnt_plan[2][0][1] = self.sim_data[:,self.t][10] + self.action_space_x[action[0]]            
            cnt_plan[2][0][2] = self.sim_data[:,self.t][11] + self.sim_data[13, self.t]*self.action_space_y[action[1]]            
            cnt_plan[2][0][3] = self.sim_data[:,self.t][12] + action[2]            
        
        else:
            cnt_plan[0][0][0] = 1
            cnt_plan[0][0][1:4] = self.sim_data[:,self.t][10:13]
            cnt_plan[2][1][0] = 1
            cnt_plan[2][1][1] = self.sim_data[:,self.t][10] + self.action_space_x[action[0]]            
            cnt_plan[2][1][2] = self.sim_data[:,self.t][11] + self.sim_data[13, self.t]*self.action_space_y[action[1]]            
            cnt_plan[2][1][3] = self.sim_data[:,self.t][12] + action[2]            
                
        step_time = int(np.round(step_time/self.delta_t,2))
        air_time = int(np.round(air_time/self.delta_t,2))
        
        self.sim_data[0:10, self.t:self.t + 2*step_time + air_time + 1], _ = \
                self.cent_mp.optimize(self.sim_data[0:10, self.t], cnt_plan, self.xt, self.w, self.ter_w, horizon)
        
        self.sim_data[10:, self.t:self.t + step_time + 1] = np.tile([self.sim_data[:,self.t][10:]],(step_time+1,1)).T #u
        self.t += step_time + air_time + 1
        self.sim_data[10, self.t:self.t + step_time] = self.sim_data[10, self.t - air_time - 1] + self.action_space_x[action[0]]
        self.sim_data[11, self.t:self.t + step_time] = self.sim_data[11, self.t - air_time - 1] + self.sim_data[13, self.t - air_time - 1]*self.action_space_y[action[1]]
        self.sim_data[12, self.t:self.t + step_time] = self.sim_data[12, self.t - air_time - 1] + action[2]
        self.sim_data[13, self.t:self.t + step_time] = -1*self.sim_data[13, self.t - air_time - 1]
        
        self.t += step_time - 1
    
        processed_state = np.zeros(8)
        processed_state[0:6] = np.take(self.sim_data[:,self.t], [0, 1, 2, 3, 4, 13])
        processed_state[0:3] -= self.sim_data[:,self.t][10:13]
        processed_state[6:] = self.v_des    
        
        if self.isdone():
            self.sim_data = self.sim_data[:,0:self.t+1]
            
        return processed_state, self.compute_cost(action), self.isdone()
    
    def isdone(self):
        '''
        This function checks if the kinematic constraints are violated
        '''
        hip = self.sim_data[:,self.t][0:3].copy()
        hip[1] -= self.sim_data[:,self.t][13]*self.b/2.0 #computing hip location
        hip[2] -= self.com_offset

        leg_length = hip - self.sim_data[:,self.t][10:13]

        if np.linalg.norm(leg_length) > self.max_leg_length:
            return True
        elif leg_length[2] < 0.05:
            return True
        else:
            return False
    
    def compute_cost(self, action):
        '''
        This function computes the cost after the step is taken
        '''
        hip = self.sim_data[:,self.t][0:3].copy()
        hip[1] -= self.sim_data[:,self.t][13]*self.b/2.0 #computing hip location
        hip[2] -= self.com_offset
        leg_length = hip - self.sim_data[:,self.t][10:13]
        
        cost = self.w_cost[0]*(np.abs(leg_length[0]) + np.abs(leg_length[1]))
        
        if self.isdone():
            cost += 100
            
        cost += self.w_cost[1]*(np.abs(self.sim_data[:,self.t][3] - self.v_des[0]) + np.abs(self.sim_data[:,self.t][4] - self.v_des[1]))
        
        cost += self.w_cost[2]*(np.abs(self.action_space_x[action[0]]))
        cost += self.w_cost[2]*np.abs(np.abs(self.action_space_y[action[1]]) - self.b)
        cost += self.w_cost[2]*(np.abs(action[2]))
        
        return np.round(cost, 2)
    
    def random_action(self):
        '''
        This function takes a random action
        '''
        action_x = np.random.randint(0, len(self.action_space_x))
        action_y = np.random.randint(0, len(self.action_space_y))
        action_z = np.random.rand(-self.max_step_ht, self.max_step_ht)
        
        return [action_x, action_y, action_z]
    