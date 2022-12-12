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
