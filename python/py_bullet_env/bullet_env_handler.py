## This file contains functions to handle environments
## Date : 29/05/2020
## Author : Avadesh Meduri

import numpy as np

# from urdf_parser_py.urdf import URDF

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

    def return_terrain_height(self, ux, uy):
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
                            



        

