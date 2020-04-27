## This a test of the SoloBulletEnv
## Author : Avadesh Meduri
## Date : 7/4/2020

from py_bullet_env.bullet_solo_env import SoloBulletEnv


kp = [100, 100, 100]
kd = [1, 1, 1]
kp_com = [100, 100, 100]
kd_com = [2, 2, 30]
kp_ang_com = [300, 300, 300]
kd_ang_com = [100, 100, 100]

step_time = 0.1
ht = 0.2
des_com = [0.2, 0, ht]
des_vel = [0, 0, 0]
x_ori = [0, 0, 0, 1]
x_angvel = [0, 0, 0]

solo_env = SoloBulletEnv(ht, step_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)
solo_env.start_recording("test")
solo_env.stop_recording()
x, xd, u = solo_env.reset_env()
state_x = [x[0] - u[0], xd[0]]
state_y = [x[1] - u[1], xd[1]]
for n in range(50):
    action = [0.2, 0]
    x, xd, u = solo_env.step_env(action, des_com, des_vel, x_ori, x_angvel)
    state_x = [x[0] - u[0], xd[0]]
    state_y = [x[1] - u[1], xd[1]]
