## This is the demo of bolt running on uneven terrain ground
## Author: Avadesh Meduri
## Date : 28/06/2020

import numpy as np
from py_bullet_env.bullet_cent_bolt_env import BulletCentBoltEnv
from py_bullet_deepq_stepper.cent_dq_stepper import CentEnv, DQStepper, predict_terrain_action_value
from py_bullet_env.bullet_env_handler import TerrainHandler

#####################################################################
kp = [0, 0, 25]
kd = [0, 0, 10]
kp_com = [0, 0, 20]
kd_com = [0, 0, 10]
kp_ang_com = [0, 0, 0]
kd_ang_com = [0, 0, 0]

step_time = 0.1
air_time = 0.1
ht = 0.3

bolt_env = BulletCentBoltEnv(ht, step_time, air_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)
###################################################################
b = 0.13
actions = [11, 9]
max_step_length = [0.5, 0.5]

env_x = CentEnv(ht, 0., max_step_length[0], [0.5, 3.0, 1.5], [actions[0],1])
dqs_x = DQStepper(env_x, lr=1e-4, gamma=0.98, use_tarnet= True, trained_model='../models/bolt/cent_running/dqs_2_str')
env_y = CentEnv(ht, b, max_step_length[1], [0.5, 3.0, 1.5], [1,actions[1]])
dqs_y = DQStepper(env_y, lr=1e-4, gamma=0.98, use_tarnet= True, trained_model='../models/bolt/cent_running/dqs_2_side')
# print(env_y.action_space_y)

##################################################################

terrain_dir = "/home/ameduri/pydevel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/terrains/run.urdf"
bolt_env.load_terrain(terrain_dir)
terr = TerrainHandler(terrain_dir)

terr_arr = np.zeros((actions[0]*actions[1],3))
terr_arr[:,0] = np.tile(np.arange(actions[0]), actions[1])
terr_arr[:,1] = np.repeat(np.arange(actions[1]), actions[0])

###################################################################
F = [0, 0, 0]
w = 0.0
no_steps = 5
des_com = [0.0, 0, ht]
des_vel = [0.0, 0.0, 0]
x_ori = [0, 0, 0, 1]
x_angvel = [0, 0, 0]

x, u, n = bolt_env.reset_env()
state = np.around([x[0] - u[0], x[1] - u[1], x[2] - u[2], x[3], x[4], n, des_vel[0], des_vel[1]],2)
bolt_env.update_gains([30, 33, 34], [1.0, 1.25, 1.35], [0, 0, 20], [0, 0, 2], [10, 10, 0], [1, 1, 0]) # dqs_2

bolt_env.start_recording("prel_running.mp4")

for i in range(no_steps):
    
    terrain = 0.00
    # for x axis
    state_x = state.copy()
    state_x[1] = 0.0
    state_x[4] = 0.0
    state_x[7] = 0.0
    action_x = dqs_x.predict_q(state_x, terrain)[1] 
    # for y axis
    state_y = state.copy()
    state_y[0] = 0.0
    state_y[3] = 0.0
    state_y[6] = 0.0
    action_y = dqs_y.predict_q(state_y, terrain)[1] 
    action = np.array([int(action_x[0]), int(action_y[1]), 0])
    
    u_x = u[0] + env_x.action_space_x[int(action[0])]
    u_y = u[1] + n*env_y.action_space_y[int(action[1])]
    u_z = u[2] + action[2]
        
    x, u_new, n = bolt_env.step_env([u_x, u_y, u_z])

    print(action, state[0:6])
    # print(u_new, [u_x, u_y, u_z])

    u = u_new
    state = np.around([x[0] - u[0], x[1] - u[1], x[2] - u[2], x[3], x[4], n, des_vel[0], des_vel[1]], 2)

# bolt_env.plot()

bolt_env.stop_recording()
