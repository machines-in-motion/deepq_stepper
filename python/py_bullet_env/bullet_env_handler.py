## This file contains functions to handle environments
## Date : 29/05/2020
## Author : Avadesh Meduri

import numpy as np
import os
import pybullet as p
from urdfpy import URDF

class TerrainHandler:

    def __init__(self, robotId):
        self.t = 0.01 # safety margin 
        self.robotId = robotId
              
    def return_terrain_height(self, ux, uy):
        '''
        This function computes height of terrain using ray tracing
        Input:
            ux : step location in x (world frame)
            uy : step location in y (world frame)
            z_foot: location of the foot in the z axis
        '''
        data = p.rayTest([ux, uy, 0.5], [ux, uy, -1])[0]
        u_z = np.round(data[3][2],2) 

        while data[0] == self.robotId:
            # prevents considering the robot bodu as ground
            data = p.rayTest([ux, uy, u_z - 0.01], [ux, uy, -1])[0]
            u_z = np.round(data[3][2],2) 

        if u_z < 0:
            u_z = 0.0

        return u_z



class TerrainGenerator:
    
    def __init__(self, dir):
        '''
        dir : the directory where the elementary shapes exist
        '''
        self.dir = dir
        self.box_arr  = os.listdir(dir + '/box/')
        self.shpere_arr  = os.listdir(dir + '/sphere/')

        self.terr_id_arr = []

    def create_box(self, location, ori, ht = None):
        '''
        This function creates a box and places it at a desried location
        Input:
            location : [x,y]
            ori : orientation
            ht : height of box (string)
        '''
        if ht :
            self.terr_id_arr.append(p.loadURDF(self.dir + '/box/' + ht + '.urdf', [0., 0., -1.]))

        else:
            box_index = np.random.randint(0, len(self.box_arr))
            self.terr_id_arr.append(p.loadURDF(self.dir + '/box/' + self.box_arr[box_index], [0., 0., -1.]))
        
        p.resetBasePositionAndOrientation(self.terr_id_arr[-1]  , location , ori)

        p.stepSimulation()


    def create_sphere(self, location, rad = None):
        '''
        This function creates a sphere and places it at a desried location
        Input:
            location : [x,y]
            ht : height of box (string)
        '''
        if rad :
            self.terr_id_arr.append(p.loadURDF(self.dir + '/sphere/' + rad + '.urdf', [0., 0., -1.]))

        else:
            sphere_index = np.random.randint(0, len(self.box_arr))
            self.terr_id_arr.append(p.loadURDF(self.dir + '/sphere/' + self.box_arr[sphere_index], [0., 0., -1.]))
        
        p.resetBasePositionAndOrientation(self.terr_id_arr[-1]  , location, [0, 0, 0, 1])

        p.stepSimulation()


    def load_terrain(self, dir):
        '''
        This function loads terrain in the simulation
        Input:
            dir : path to urdf
        '''
        terrain = (dir)
        terrain_id = p.loadURDF(terrain)

        p.stepSimulation()


