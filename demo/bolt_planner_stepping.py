## demo of blot stepping on flat ground with dq stepper
## Date : 28/05/2020
## Author : Avadesh Meduri

import numpy as np
from py_deepq_stepper.twod_dq_stepper import TwoDQStepper, TwoDLipmEnv
from py_bullet_env.bullet_bolt_env import BoltBulletEnv
from py_bullet_env.bullet_env_handler import TerrainHandler
#####################################################################
kp = [35, 35, 25]
kd = [10, 10, 10]
kp_com = [0, 0, 20]
kd_com = [0, 0, 10]
kp_ang_com = [0, 0, 0]
kd_ang_com = [0, 0, 0]

step_time = 0.1
stance_time = 0.0
ht = 0.2

terrain_dir = "/home/ameduri/pydevel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/terrains/plates.urdf"
bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)
bolt_env.load_terrain(terrain_dir)
terr = TerrainHandler(terrain_dir)
##################################################################
env = TwoDLipmEnv(0.2, 0.13, 0.22, [1, 3, 0], no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

dqs = TwoDQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model='../models/twod/dqs_2')        
###################################################################
# stepping with regularizing
q_mat = np.zeros((no_actions[1], no_actions[0]))
reg_mat = np.zeros((no_actions[1], no_actions[0]))
for y in range(reg_mat.shape[0]):
    for x in range(reg_mat.shape[1]):
        reg_mat[y][x] = np.linalg.norm([env.action_space_x[x], env.action_space_y[y] - env.b])

F = [0, 0, 0]
w = 0.0
no_steps = 26
des_com = [0.0, 0, ht]
des_vel = [1.0, 0.0, 0]
x_ori = [0, 0, 0, 1]
x_angvel = [0, 0, 0]

x, xd, u, n = bolt_env.reset_env()
state = [x[0] - u[0], x[1] - u[1], xd[0], xd[1], n, des_vel[0], des_vel[1]]
bolt_env.update_gains([40, 40, 40], [10, 10, 10], [0, 0, 20], [0, 0, 10], [50, 50, 0], [40, 40, 0])
xd_arr = []
# bolt_env.start_recording("terrain_stepping.mp4")
for i in range(no_steps):
    q = dqs.predict_q(state)
    # if i > 15 and i < 18:
    #     F = [-4, 0, 0]
    # else:
    #     F = [0, 0, 0]
    for i in range(len(q)):
        q_mat[int(dqs.x_in[i,8]), int(dqs.x_in[i,7])] = q[i]
        u_y = n*env.action_space_y[int(dqs.x_in[i,8])] + u[1]
        u_x = env.action_space_x[int(dqs.x_in[i,7])] + u[0]
        # tmp = terr.check_terrain(u_x, u_y)
        q_mat[int(dqs.x_in[i,8]), int(dqs.x_in[i,7])] += terr.check_terrain(u_x, u_y)

    q_mat = q_mat + w*reg_mat
    
    action = np.unravel_index(q_mat.argmin(), q_mat.shape)
    u_x = env.action_space_x[action[1]] + u[0]
    u_y = n*env.action_space_y[action[0]] + u[1]
    x, xd, u_new, n = bolt_env.step_env([u_x, u_y], des_com, des_vel, x_ori, x_angvel, F)
    xd_arr.append(xd[0])
    u = u_new
    state = [x[0] - u[0], x[1] - u[1], xd[0], xd[1], n, des_vel[0], des_vel[1]]

# bolt_env.stop_recording()

print(np.average(xd_arr))