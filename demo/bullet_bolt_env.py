# This is a test of bullet bolt env
## Author : Avadesh Meduri
## Date : 5/5/2020

from py_bullet_env.bullet_bolt_env import BoltBulletEnv

kp = [50, 50, 50]
kd = [1, 1, 10]
kp_com = [10, 0, 0]
kd_com = [10, 0, 0]
kp_ang_com = [10, 10, 10]
kd_ang_com = [1, 1, 1]

step_time = 0.1
stance_time = 0.01
ht = 0.2
des_com = [0.0, 0, ht]
des_vel = [0, 0, 0]
x_ori = [0, 0, 1, 0]
x_angvel = [0, 0, 0]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)

x, xd, u, n = bolt_env.reset_env()

state = [x[0] - u[0], x[1] - u[1], xd[0], xd[1], n]
# print(state)
# print("init")
# bolt_env.update_gains([0, 0, 0], kd, kp_com, kd_com, kp_ang_com, kd_ang_com)
for i in range(50):
    if n > 0:
        action = [0.0, 0.07]
    if n < 0:
        action = [0.0, -0.07]
    x, xd, u, n = bolt_env.step_env(action, des_com, des_vel, x_ori, x_angvel)
    # print(u, n)
    state = [x[0] - u[0], x[1] - u[1], xd[0], xd[1], n]
