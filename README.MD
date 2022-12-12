# DeepQ stepper

This github repo contains the software used to develop the DeepQ stepper algorithm. 


## What is the DeepQ Steper : 

The DeepQ stepper, is a reactive bipedal stepping framework that can approximately
learn the 3D capture regions of both simplified and full robot
dynamic models using reinforcement learning, which can then
be used to find optimal steps. The stepper can take into
account the entire dynamics of the robot, ignored in most
reactive steppers, leading to a significant improvement in
performance. The DeepQ stepper can handle nonconvex terrain
with obstacles, walk on restricted surfaces like stepping stones
while tracking different velocities, and recover from external
disturbances for a constant low computational cost.

## Dependencies :
1. pytorch
2. pybullet
3. robot_properties_bolt - https://github.com/open-dynamic-robot-initiative/robot_properties_bolt

## Using the code : 

1. The demo folder contains various examples/results that demonstrate the capability of the DeepQ stepper. All the results are shown on the biped robot Bolt, which has 12 Degrees of Freedom. 
2. The code used to train the DeepQ stepper, the implementation of the pybullet environment of Bolt used to deploy and train the DeepQ stepper are available in the python directory. Also, the 2D and 3D LiPM model implementations are also in this directory. 
3. The motion planner that generated the Center of Mass trajectory given the foot step plan is also in the python/py_motion_planner directory. These trajectories are tracked using the impedance controller. 


## Citation 
```
@inproceedings{meduri2021deepq,
  title={DeepQ Stepper: A framework for reactive dynamic walking on uneven terrain},
  author={Meduri, Avadesh and Khadiv, Majid and Righetti, Ludovic},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2099--2105},
  year={2021},
  organization={IEEE}
}

```
