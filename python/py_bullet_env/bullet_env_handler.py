## This file contains functions to handle environments
## Date : 29/05/2020
## Author : Avadesh Meduri

import numpy as np

import pybullet as p
from urdfpy import URDF

class TerrainHandler:

    def __init__(self, dir):
        self.terrain = URDF.load(dir)
        self.t = 0.01 # safety margin 

    def check_terrain(self, ux, uy):
        '''
        Checks the height of the terrain for the given step
        Input:
            ux : step location in x (world frame)
            uy : step location in y (world frame)
        '''
        flag = 0
        for link in self.terrain.links:
            name = link.name
            origin_xyz = np.array(link.visuals[0].origin)[:,3][0:3]
            geometry = np.array(link.visuals[0].geometry.box.size)
            if ux > origin_xyz[0] - 0.5*geometry[0] + self.t:
                if ux < origin_xyz[0] + 0.5*geometry[0] - self.t:
                    if uy > origin_xyz[1] - 0.5*geometry[1] + self.t:
                        if uy < origin_xyz[1] + 0.5*geometry[1] - self.t:
                           flag = 1
                           break

        if flag == 1:
            return 0
        else: 
            return 999999

    def return_terrain_height_old(self, ux, uy):
        '''
        Returns the height of the terrain for the given step location
        Input:
            ux : step location in x (world frame)
            uy : step location in y (world frame)
        
        '''
        ht = 9999999
        for link in self.terrain.links:
            name = link.name
            origin_xyz = np.array(link.visuals[0].origin)[:,3][0:3]
            geometry = np.array(link.visuals[0].geometry.box.size)
            if ux > origin_xyz[0] - 0.5*geometry[0] + self.t:
                if ux < origin_xyz[0] + 0.5*geometry[0] - self.t:
                    if uy > origin_xyz[1] - 0.5*geometry[1] + self.t:
                        if uy < origin_xyz[1] + 0.5*geometry[1] - self.t:
                           ht = geometry[2]
                           break

        return ht
                  
    def return_terrain_height(self, ux, uy, z_foot):
        '''
        This function computes height of terrain using ray tracing
        Input:
            ux : step location in x (world frame)
            uy : step location in y (world frame)
            z_foot: location of the foot in the z axis
        '''
        u_z = np.around(p.rayTest([ux, uy, 0.06 + z_foot], [ux, uy, -1])[0][3][2], 2)
        if u_z > 0.07:
            u_z = 0.07
        return u_z


        



        

