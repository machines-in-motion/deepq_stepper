# This file contains code to train deepq stepper directly in pybullet with bullet_env
# Author : Avadesh Meduri
# Date : 6/08/2020

import numpy as np
from py_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv
from py_bullet_env.bullet_bolt_env import BoltBulletEnv
import time

kp = [100, 100, 150]
kd = [15, 15, 15]
kp_com = [0, 0, 150]
kd_com = [0, 0, 10]
kp_ang_com = [100, 100, 0]
kd_ang_com = [20, 20, 0]

step_time = 0.1
stance_time = 0.03
ht = 0.28
w = [0.5, 3, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, [1, 3, 0], no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions)

###################################################################
F = [0, 0, 0]
terrain = np.zeros(no_actions[0]*no_actions[1])

no_epi = 1
no_steps = 10

##################################################################
for e in range(no_epi):

    v_init = [2*(np.random.rand() - 0.5), (np.random.rand() - 0.5)]
    v_des = [0.25*np.random.randint(-4, 5), 0.25*np.random.randint(-2, 3)]
    x, xd, u, n = bolt_env.reset_env([0, 0, ht, v_init[0], v_init[1]])
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]
    epi_cost = 0
    
    for i in range(no_steps):

        action = dqs.predict_q(state, terrain)[1]
        
        u_x = env.action_space_x[int(action[0])] + u[0]
        u_y = n*env.action_space_y[int(action[1])] + u[1]
        u_z = action[2] + u[2]
        x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
        next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
        state = next_state
        u = u_new
        epi_cost += cost
        if done:
            break

# bolt_env.plot()