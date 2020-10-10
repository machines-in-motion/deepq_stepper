## This file contains DQstepper class that trains for slow step times using warm start greedy training
## Author : Avadesh Meduri
## Date : 9/10/2020

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from matplotlib import pyplot as plt


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
    def __init__(self, no_actions = [9, 7], lr = 1e-4, gamma = 0.9, use_tarnet = False, env = None,\
                 trained_model = None, wm_model_x = None, wm_model_y = None, wm_model_xy = None):
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
        
        self.use_wm_model_x = False
        self.use_wm_model_y = False
        self.use_wm_model_xy = False

        if env:
            self.env = env
            # used in epsillon greedy for stepping in y direction
        if wm_model_y:
            self.use_wm_model_y = True
            self.dq_wm_y = NN(11,1).to(self.device)
            self.dq_wm_y.load_state_dict(torch.load(wm_model_y))
            self.dq_wm_y.eval()

            self.y_in = np.zeros((1*self.no_actions[1], 11))
            self.y_in[:,9] = np.repeat(np.arange(self.no_actions[1]), 1)

        if wm_model_x:
            self.use_wm_model_x = True
            self.dq_wm_x = NN(11,1).to(self.device)
            self.dq_wm_x.load_state_dict(torch.load(wm_model_x))
            self.dq_wm_x.eval()

            self.wm_x_in = np.zeros((self.no_actions[0]*1, 11))
            self.wm_x_in[:,8] = np.tile(np.arange(self.no_actions[0]), 1)

        if wm_model_xy:
            self.use_wm_model_xy = True
            self.use_wm_model_x = False
            self.use_wm_model_y = False
            self.dq_wm_xy = NN(11,1).to(self.device)
            self.dq_wm_xy.load_state_dict(torch.load(wm_model_xy))
            self.dq_wm_xy.eval()

            self.xy_in = np.zeros((self.no_actions[0]*self.no_actions[1], 11))
            self.xy_in[:,8] = np.tile(np.arange(self.no_actions[0]), self.no_actions[1])
            self.xy_in[:,9] = np.repeat(np.arange(self.no_actions[1]), self.no_actions[0])


        assert self.use_wm_model_xy*self.use_wm_model_x == 0
        assert self.use_wm_model_xy*self.use_wm_model_y == 0
        

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
    
    def predict_eps_wm(self, x, eps = 0.1):
        if np.random.random() > eps:
            return self.predict_action_value(x)[0]
        else:
#             action_x = np.random.randint(self.no_actions[0])
#             action_y = np.random.randint(self.no_actions[1])
            action_z = 0
            
            if self.use_wm_model_xy:
                self.xy_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = x
                torch_xy_in = torch.FloatTensor(self.xy_in, device = self.device)
                with torch.no_grad():
                    q_values = self.wm_dq_xy(torch_xy_in).detach().numpy()
                    action_index = np.argmin(q_values)
                    action_x = int(action_index%self.no_actions[0])
                    action_y = int(action_index//self.no_actions[0])
                
            if self.use_wm_model_y:
                y_in = x.copy()
                y_in[0] = 0
                y_in[3] = 0
                self.y_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = y_in
                torch_y_in = torch.FloatTensor(self.y_in, device = self.device)
                with torch.no_grad():
                    q_values = self.dq_wm_y(torch_y_in).detach().numpy()
                    action_index = np.argmin(q_values)
                    action_y = int(action_index//1)
            
            if self.use_wm_model_x:
                x_in = x.copy()
                x_in[1] = 0.0
                x_in[4] = 0.0
                self.wm_x_in[:,[0, 1, 2, 3, 4, 5, 6, 7]] = x_in
                torch_x_in = torch.FloatTensor(self.wm_x_in, device = self.device)
                with torch.no_grad():
                    q_values = self.dq_wm_x(torch_x_in).detach().numpy()
                    action_index = np.argmin(q_values)
                    action_x = int(action_index)
                        
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
    
    def live_plot(self, history, e, figsize=(15,25), window = 500, title='history', show = False):
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax[0].plot(history['epi_cost'], label='epi_cost', color = 'orange')
        ax[0].grid(True)
        ax[0].legend() 
        if e > window:
            ax[1].plot(np.arange(e-window+1, e), history['epi_cost'][e-window:], label='epi_cost zoom')
            ax[1].grid(True)
            ax[1].legend() 
        ax[2].plot(history['loss'], label='loss', color = 'black')
        ax[2].grid(True)
        ax[2].legend() 
        ax[2].set_ylim(0, 60)
        plt.xlabel('episode')
        if show:
            plt.show()
        else:    
            plt.savefig('../../results_paper/dqs_1.png')
            plt.close()