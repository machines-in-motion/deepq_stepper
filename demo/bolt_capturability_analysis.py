## This file shows the analysis to understand the capture region the stepper learns
## Author: Avadesh Meduri
## Date : 19/08/2020


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
off = 0.03
w = [0.5, 3.0, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

def plot_heatmap(q_val, dqs, env):
    n = dqs.no_actions
    q_mat = np.zeros((n[1], n[0]))
    for i in range(len(q_val)):
        q_mat[int(dqs.x_in[i,9]), int(dqs.x_in[i,8])] = q_val[i]
    fig, ax = plt.subplots()
    ax.set_xticklabels(env.action_space_x[::2])
    ax.set_yticklabels(env.action_space_y)
    heatmap = ax.pcolor(q_mat, cmap='PuBu_r')
    fig.colorbar(heatmap, ax=ax)
    plt.show()


###################################################################

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/dqs_2")


##################################################################

terr_gen = TerrainGenerator('/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/shapes')

terr = TerrainHandler(bolt_env.robot.robotId)

max_length = 0.5
terr_gen.create_random_terrain(50, max_length)

# terr_gen.load_terrain("../python/py_bullet_env/terrains/stepping_stones.urdf")

###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_steps = 10
no_epi = 15
##################################################################



bolt_env.start_recording("3d_capturability_analysis.mp4")

for e in range(no_epi):

    option_number = 0
    opt = np.random.randint(4, 10)
    step_number = np.random.randint(0,2)
    v_init = np.round([2.0*(np.random.rand() - 0.5), 1.5*(np.random.rand() - 0.5)],2)
    v_des = [0, 0]
    x_init = 0.5*np.round([random.uniform(-max_length, max_length), random.uniform(-max_length, max_length)], 2)
    
    if np.power(-1, step_number) > 0:
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw= np.sign(v_init[0])*140, cameraPitch= -35, cameraTargetPosition=[x_init[0],x_init[1],0])
    else:
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw= np.sign(v_init[0])*50, cameraPitch= -35, cameraTargetPosition=[x_init[0],x_init[1],0])
    
    x, xd, u, n = bolt_env.reset_env([x_init[0], x_init[1], ht + off, v_init[0], v_init[1]])
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]    
    epi_cost = 0


    for i in range(no_steps):
        
        terrain = dqs.x_in[:,7:].copy()
        counter = 0
        for k in range(len(dqs.x_in)):
            u_x = env.action_space_x[int(dqs.x_in[k,8])] + u[0]
            u_y = n*env.action_space_y[int(dqs.x_in[k,9])] + u[1]
            u_z = terr.return_terrain_height(u_x, u_y)
            terrain[k,3] = np.around(u_z - u[2], 2)
        q_values, _ = dqs.predict_q(state, terrain[:,3])
        terrain[:,0] = np.reshape(q_values, (len(terrain[:,0],)))
        terrain = np.round(terrain[terrain[:,0].argsort()], 2)
        
        if i == step_number:
            option_number = opt

        ## This intentionally chooses the nth best action
        for action_index in range(len(terrain[:,0])):
            if action_index < option_number:
                del_u_x = env.action_space_x[int(terrain[action_index][1])] + u[0]
                del_u_y = n*env.action_space_y[int(terrain[action_index][2])] + u[1]
                del_u_z = terrain[action_index][3]
                if action_index == 0:
                    p.addUserDebugLine([del_u_x, del_u_y, del_u_z],[del_u_x, del_u_y, del_u_z + 0.5],[1, 1, 0], 3)
                else:
                    p.addUserDebugLine([del_u_x, del_u_y, del_u_z],[del_u_x, del_u_y, del_u_z + 0.5],[0, 5*action_index/20,1], 3)
                terrain = np.delete(terrain, action_index, 0)
            else:
                action = terrain[action_index][1:]
                u_x = env.action_space_x[int(action[0])] + u[0]
                u_y = n*env.action_space_y[int(action[1])] + u[1]
                u_z = action[2] + u[2]
                p.addUserDebugLine([u_x, u_y, u_z],[u_x, u_y, u_z + 0.5],[1, 0,1], 3)

                option_number = 0
                break

        if i == step_number:
            time.sleep(1.5)
            # plot_heatmap(q_values, dqs, env)

                    
        x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
        next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
        state = next_state
        u = u_new
        epi_cost += cost
        
        
        p.removeAllUserDebugItems()

bolt_env.stop_recording()