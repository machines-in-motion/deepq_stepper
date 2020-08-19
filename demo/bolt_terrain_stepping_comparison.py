## This demo runs different trained steppers and plots performance (epi cost)
## Author : Avadesh Meduri
## Date : 10/08/2020

import numpy as np
import rospkg
import pybullet as p
import time

from py_bullet_env.bullet_env_handler import TerrainHandler, TerrainGenerator

from py_bullet_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv, Buffer
from py_bullet_env.bullet_bolt_env import BoltBulletEnv

import time
from matplotlib import pyplot as plt
import random

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
off = 0.02
w = [0.5, 3.5, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################


dqs_1 = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model='../models/bolt/lipm_walking/dqs_3')

dqs_2 = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/dqs_2")

# dqs_3 = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
#     no_actions= no_actions, trained_model = "../models/dqs_3")


dqs_arr = [dqs_1, dqs_2]
dqs_ct = 0

##################################################################

terr_gen = TerrainGenerator('/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/shapes')


terr = TerrainHandler(bolt_env.robot.robotId)

max_length = 1.0
terr_gen.create_random_terrain(350, max_length)
# terr_gen.load_terrain("../python/py_bullet_env/terrains/push.urdf")

###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_steps = 15
no_epi = 50
##################################################################

history = [[], []]
e = 0
while e < no_epi:
    v_init = np.round([1.5*(np.random.rand() - 0.5), 1*(np.random.rand() - 0.5)],2)
    v_des = [0.5*random.randint(-0, 1), 0.5*random.randint(-1, 1)]
    x_init = 0.5*np.round([random.uniform(-max_length, max_length), random.uniform(-max_length, max_length)], 2)
    done_arr = []

    for j in range(len(dqs_arr)):
        x, xd, u, n = bolt_env.reset_env([x_init[0], x_init[1], ht + off, v_init[0], v_init[1]])
        state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]    
        dqs = dqs_arr[j]
        epi_cost = 0
        for i in range(no_steps):
            
            terrain = dqs.x_in[:,8:].copy()
            for i in range(len(dqs.x_in)):
                u_x = env.action_space_x[int(dqs.x_in[i,8])] + u[0]
                u_y = n*env.action_space_y[int(dqs.x_in[i,9])] + u[1]
                u_z = terr.return_terrain_height(u_x, u_y)
                terrain[i,2] = np.around(u_z - u[2], 2)
            q_values, _ = dqs.predict_q(state, terrain[:,2])
         
            action_index = np.argmin(q_values)
            action = terrain[action_index]

            u_x = env.action_space_x[int(action[0])] + u[0]
            u_y = n*env.action_space_y[int(action[1])] + u[1]
            u_z = action[2] + u[2]
            # print(u_z)
            x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
            next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
            state = next_state
            u = u_new
            epi_cost += cost
            
            if done:
                # print('terminated ..')
                break        

        done_arr.append(epi_cost)
    
    if (np.array(done_arr) < 100).all(): 
        for k in range(len(done_arr)):
            history[k].append(done_arr[k])
        
        e += 1
        

print('Bullet dqs win percentage : ' + str(np.sum(np.greater(history[0], history[1]))/no_epi))

# plt.title("Comparison: Episode cost for stepper trained in different environments")
# plt.title("Comparison: Episode cost for stepper trained for 2d and 3d, but tested in 3d")
plt.plot(history[0], label = 'IPM Env')
# plt.plot(history[0], label = 'bullet_2d_dqs')
plt.plot(history[1], label = 'Bullet Env')
plt.ylabel('Episode Cost')
plt.xlabel('Episode Number')
plt.legend()
plt.grid()
plt.show()

