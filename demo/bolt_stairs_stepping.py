## demo of blot stepping on 3d terrain with dq stepper
## Date : 4/06/2020
## Author : Avadesh Meduri

import numpy as np
from py_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv
from py_bullet_env.bullet_bolt_env import BoltBulletEnv
from py_bullet_env.bullet_env_handler import TerrainHandler

import pybullet as p
#####################################################################
kp = [35, 35, 25]
kd = [10, 10, 10]
kp_com = [0, 0, 20]
kd_com = [0, 0, 10]
kp_ang_com = [0, 0, 0]
kd_ang_com = [0, 0, 0]

step_time = 0.1
stance_time = 0.03
ht = 0.28

terrain_dir = "/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/terrains/stairs.urdf"
bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)
bolt_env.load_terrain(terrain_dir)
terr = TerrainHandler(terrain_dir)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, [1, 3, 0], no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model='../models/bolt/dqs_3')        
###################################################################
F = [0, 0, 0]
w = 0.0
no_steps = 80
des_vel = [0.5, 0.0, 0]

x, xd, u, n = bolt_env.reset_env()
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, des_vel[0], des_vel[1]]
bolt_env.update_gains([25, 25, 25], [7, 7, 10], [0, 0, 20], [0, 0, 10], [50, 50, 0], [50, 50, 0])
xd_arr = []
bolt_env.start_recording("3d_stairs_stepping.mp4")
for i in range(no_steps):

    terrain = dqs.x_in[:,8:].copy()
    p.resetDebugVisualizerCamera( cameraDistance=0.9, cameraYaw=120, cameraPitch=-25, cameraTargetPosition=[1.6, 0,0])

    if i < 30:
        p.resetDebugVisualizerCamera( cameraDistance=0.9, cameraYaw=120, cameraPitch=-25, cameraTargetPosition=[0.5, 0,0])

    elif i < 50:
        p.resetDebugVisualizerCamera( cameraDistance=0.9, cameraYaw=150, cameraPitch=-25, cameraTargetPosition=[0.9, 0,0])
    
    elif i < 65:
        p.resetDebugVisualizerCamera( cameraDistance=0.9, cameraYaw=150, cameraPitch=-25, cameraTargetPosition=[1.2, 0,0])

    if i > 70:
        des_vel = [0, 0, 0]

    for i in range(len(dqs.x_in)):
        u_y = n*env.action_space_y[int(dqs.x_in[i,9])] + u[1]
        u_x = env.action_space_x[int(dqs.x_in[i,8])] + u[0]
        u_z = terr.return_terrain_height(u_x, u_y, u[2])
        terrain[i,2] = np.around(u_z - u[2], 2)

    
    q_values, _ = dqs.predict_q(state, terrain[:,2])
    while True:
        action_index = np.argmin(q_values)
        if terrain[action_index, 2] < 100:
            action = terrain[action_index]
            break
        else:
            np.delete(terrain, action_index)
    
    u_x = env.action_space_x[int(action[0])] + u[0]
    u_y = n*env.action_space_y[int(action[1])] + u[1]
    u_z = action[2] + u[2]
    # print(u_x, u_y, u_z)
    x, xd, u_new, n = bolt_env.step_env([u_x, u_y, u_z], des_vel, F)
    xd_arr.append(xd[0])
    u = u_new
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, des_vel[0], des_vel[1]]

bolt_env.stop_recording()
bolt_env.plot()
print(np.average(xd_arr))