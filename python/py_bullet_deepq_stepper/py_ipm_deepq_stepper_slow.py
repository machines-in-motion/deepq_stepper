## This code trains deepQ stepper in bullet directly for slower step times using warm starting and
## priorized sampling
## Author : Avadesh Meduri
## Date : 9/10/2020

import numpy as np
import random
from py_bullet_deepq_stepper.dq_stepper import InvertedPendulumEnv, Buffer
from py_bullet_deepq_stepper.dq_stepper_slow import DQStepper

from py_bullet_env.bullet_bolt_env import BoltBulletEnv

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

kp = [30, 30, 30]#[150, 150, 150]
kd = [5, 5, 5] #[15, 15, 15]
kp_com = [0, 0, 30] #[0, 0, 150.0]
kd_com = [0, 0, 10] #[0, 0, 20.0]
kp_ang_com = [40, 40, 0]#[100, 100, 0]
kd_ang_com = [5, 5, 0]#[25, 25, 0]

F = [0, 0, 0]

step_time = 0.2
stance_time = 0.00
ht = 0.35
off = 0.0
w = [0.5, 3, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, [0.4, 0.3], w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################

name = 'dqs_1_light'
dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, no_actions= no_actions, env = env, \
                wm_model_x="../../models/dqs_x11", wm_model_y="../../models/dqs_y9", trained_model = "../../models/dqs_11_9_2")


###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])

e = 1
no_epi = 40000
no_steps = 10

buffer_size = 2000 #6000
buffer = Buffer(buffer_size)
batch_size = 16
epsillon = 0.2 #0.8
ratio = 0.8

##################################################################
history = {'loss':[], 'epi_cost':[]}
while e < no_epi:

    v_init = np.round([0.5*random.uniform(-1.0, 1.0), 0.5*random.uniform(-1.0, 1.0)], 2)
    v_des = [0.5*random.randint(-1, 1), 0.5*random.randint(-1, 1)]
    x, xd, u, n = bolt_env.reset_env([0, 0, ht + off, v_init[0], v_init[1]])
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]

    if e == 2:
        epsillon = 0.2 #0.5
        print("updated epsillon ...")

    if e % 500 == 0 and e > 1:
        print('saving model ...')
        torch.save(dqs.dq_stepper.state_dict(), "../../models/" + name)
        dqs.live_plot(history, e)
        #reducing epsillon
        if e % 1000 == 0 and e > 5000:
            epsillon = epsillon/2
            
    if buffer.size() % 1000 == 0:    
        if buffer.size() == buffer_size:
            history['epi_cost'].append(epi_cost)
            history['loss'].append(loss)
            assert len(history['loss']) == len(history['epi_cost'])
            if e == 1:
                print("Buffer Done. Training started ...")
            e += 1
        else:
            print(buffer.size())
    
    epi_cost = 0
    for i in range(no_steps):

        action = dqs.predict_eps_greedy(state, epsillon)
        # action = dqs.predict_eps_wm(state, epsillon)
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
            batch = np.concatenate((buffer.sample(int(ratio*batch_size)), buffer.buffer[-int((1-ratio)*batch_size)-1:]))
            loss = dqs.optimize(batch, tau = 0.001) 
            epi_cost += cost
        
        if done:
            break

torch.save(dqs.dq_stepper.state_dict(), "../../models/" + name)

