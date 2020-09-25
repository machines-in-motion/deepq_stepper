## This file trains the dq stepper in bullet environment with terrain (3d)
## Author : Avadesh Meduri
## Date : 12/08/2020

import numpy as np
import random
from py_bullet_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv, Buffer
from py_bullet_env.bullet_bolt_env import BoltBulletEnv
from py_bullet_env.bullet_env_handler import TerrainHandler, TerrainGenerator

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

kp = [150, 150, 150]
kd = [15, 15, 15]
kp_com = [0, 0, 150.0]
kd_com = [0, 0, 20.0]
kp_ang_com = [100, 100, 0]
kd_ang_com = [25, 25, 0]

F = [0, 0, 0]

step_time = 0.1
stance_time = 0.02
ht = 0.28
off = 0.0
w = [0.5, 3.0, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################

name = 'dqs_1'
dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions)

###################################################################
e = 1
no_epi = 15000
no_steps = 10

buffer_size = 7000
buffer = Buffer(buffer_size)
batch_size = 16
epsillon = 0.2

##################################################################
terr_gen = TerrainGenerator('/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/shapes')

max_length = 1.0

terr = TerrainHandler(bolt_env.robot.robotId)
terr_gen.create_random_terrain(40, max_length)

##################################################################
history = {'loss':[], 'epi_cost':[]}
while e < no_epi:

    v_init = np.round([random.uniform(-1., 1.), random.uniform(-1, 1)], 2)
    v_des = [0.5*random.randint(-1, 1), 0.5*random.randint(-1, 1)]
    x_init = 0.8*np.round([random.uniform(-max_length, max_length), random.uniform(-max_length, max_length)], 2)
    x, xd, u, n = bolt_env.reset_env([x_init[0], x_init[1], ht + off, v_init[0], v_init[1]])
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]
    if e % 500 == 0 and e > 10:
        # print('saving model ...')
        torch.save(dqs.dq_stepper.state_dict(), "../../models/" + name)
        dqs.live_plot(history, e)
        terr_gen.create_random_terrain(3, max_length)

        #reducing epsillon
        if e%1000 == 0 and e > 5000:
            epsillon = epsillon/2
            
    if buffer.size() % 1000 == 0:    
        if buffer.size() == buffer_size:
            history['epi_cost'].append(epi_cost)
            history['loss'].append(loss)
            assert len(history['loss']) == len(history['epi_cost'])
            e += 1
        else:
            print(buffer.size())

    epi_cost = 0
    for i in range(no_steps):

        terrain = dqs.x_in[:,8:].copy()
        if random.uniform(0, 1) > epsillon:
            for i in range(len(dqs.x_in)):
                u_x = env.action_space_x[int(dqs.x_in[i,8])] + u[0]
                u_y = n*env.action_space_y[int(dqs.x_in[i,9])] + u[1]
                u_z = terr.return_terrain_height(u_x, u_y)
                terrain[i,2] = np.around(u_z - u[2], 2)

            q_values, _ = dqs.predict_q(state, terrain[:,2])
            action_index = np.argmin(q_values)
            action = terrain[action_index]

        else:
            action = [random.randint(0, no_actions[0] - 1), random.randint(0, no_actions[1] - 1), 0.0]
            u_x = env.action_space_x[action[0]] + u[0]
            u_y = n*env.action_space_y[action[1]] + u[1]
            u_z = terr.return_terrain_height(u_x, u_y)
            action[2] = np.around(u_z - u[2], 2)
            
        u_x = env.action_space_x[int(action[0])] + u[0]
        u_y = n*env.action_space_y[int(action[1])] + u[1]
        u_z = action[2] + u[2]
            
        x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
        next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
        buffer.store(state, action, cost, next_state, done)
        state = next_state
        u = u_new
        if buffer.size() == buffer_size:
            ## optimizing DQN
            loss = dqs.optimize(buffer.sample(batch_size)) 
            epi_cost += cost
        
        if done:
            break

torch.save(dqs.dq_stepper.state_dict(), "../../models/" + name)

# bolt_env.plot()