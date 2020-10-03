## demo of bolt stepping on 3d terrain with dq stepper along with push recovery
## this demo shows that the dq stepper learns capturability in 3d
## Date : 3/07/2020
## Author : Avadesh Meduri

import numpy as np
from py_bullet_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv
from py_bullet_env.bullet_bolt_env import BoltBulletEnv
from py_bullet_env.bullet_env_handler import TerrainHandler, TerrainGenerator

import pybullet as p
#####################################################################
kp = [50, 50, 50]
kd = [5, 5, 5]
kp_com = [0, 0, 130.0]
kd_com = [0, 0, 20.0]
kp_ang_com = [100, 100, 0]
kd_ang_com = [25, 25, 0]

F = [0, 0, 0]

step_time = 0.1
stance_time = 0.02
ht = 0.28
off = 0.02
w = [0.5, 3.0, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
terr = TerrainHandler(bolt_env.robot.robotId)
terr_gen = TerrainGenerator('/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/shapes')

max_length = 0.5

terr_gen.create_random_terrain(20, max_length)

##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, [1, 3, 0], no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model='../models/bolt/lipm_walking/dqs_3')

# dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
#     no_actions= no_actions, trained_model='../models/bolt/bullet_walking/dqs_2')        
###################################################################
F = [0, 0, 0]
w = 0.0
no_steps = 1000
des_vel = [0.0, 0.0, 0]

v_init = [0.0, 0]
v_des = [0.0, 0]
x_init = [0.0, 0]

x, xd, u, n = bolt_env.reset_env([x_init[0], x_init[1], ht + off, v_init[0], v_init[1]])
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]    
epi_cost = 0

p.resetDebugVisualizerCamera( cameraDistance=0.7, cameraYaw=0, cameraPitch=-25, cameraTargetPosition=[0.0, 0,0])

bolt_env.start_recording("3d_push_recovery.mp4")
for i in range(no_steps):


    if i < 7 and i > 2:
        F = [3.0, 0, 0]
        
    elif i > 12 and i < 17 :
        F = [0, 3.2, 0]
        p.resetDebugVisualizerCamera( cameraDistance=0.7, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=[0.3, 0,0])

    elif i > 20 and i < 25 :
        F = [0, -3, 0]
        p.resetDebugVisualizerCamera( cameraDistance=0.7, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=[0.3, 0,0])

    elif i > 30 and i < 35 :
        F = [-3, 0, 0]
        p.resetDebugVisualizerCamera( cameraDistance=0.7, cameraYaw=0, cameraPitch=-25, cameraTargetPosition=[0.0, 0,0])

    else: 
        F = [0, 0, 0]


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
    
    # if done:
    #     print('terminated ..')
    #     break        

bolt_env.stop_recording()
bolt_env.plot()
