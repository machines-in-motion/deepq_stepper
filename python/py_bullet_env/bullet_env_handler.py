## This file contains functions to handle environments
## Date : 29/05/2020
## Author : Avadesh Meduri

import numpy as np
import random
import os
import pybullet as p
from urdfpy import URDF

class TerrainHandler:

    def __init__(self, robotId, margin = 0.015):
        self.t = 0.01 # safety margin 
        self.robotId = robotId
        self.margin = margin
    
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
          
    def batch_return_terrain_height(self, ux, uy):
        '''
        This function computes height of terrain using batch ray tracing
        Input:
            ux : step location in x (world frame)
            uy : step location in y (world frame)
            z_foot: location of the foot in the z axis
        '''
        margin = self.margin
        # data_single = p.rayTest([ux, uy, 0.5], [ux, uy, -1])[0]
        data = p.rayTestBatch([[ux, uy, 0.5], [ux + margin, uy, 0.5], [ux - margin, uy, 0.5], [ux, uy + margin, 0.5], [ux, uy - margin, 0.5]], \
            [[ux, uy, -0.5], [ux + margin, uy, -0.5], [ux - margin, uy, -0.5], [ux, uy + margin, -0.5], [ux, uy - margin, -0.5]])


        id_data = np.array(data, dtype = object)[:,0]
        terr_data = np.array(data, dtype = object)[:,3]
        u_z = 1000
        for i in range(len(data)):
            if u_z > np.round(terr_data[i][2], 2):
                u_z = np.round(terr_data[i][2], 2)

        while id_data.any() == self.robotId:
            # prevents considering the robot bodu as ground
            data = p.rayTestBatch([[ux, uy, u_z - 0.01], [ux + margin, uy, u_z - 0.01], [ux - margin, uy, u_z - 0.01], [ux, uy + margin, u_z - 0.01], [ux, uy - margin, u_z - 0.01]], \
            [[ux, uy, -0.5], [ux + margin, uy, -0.5], [ux - margin, uy, -0.5], [ux, uy + margin, -0.5], [ux, uy - margin, -0.5]])
            
            id_data = np.array(data, dtype = object)[:,0]
            terr_data = np.array(data, dtype = object)[:,3]
            u_z = 1000
            for i in range(len(data)):
                if u_z > np.round(terr_data[i][2], 2):
                    u_z = np.round(terr_data[i][2], 2)

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

    def create_random_terrain(self, n, max_length):
        '''
        This function creates a random terrain of spheres and boxes
        Input:
            n : number of objects to be placed
            max_length : max distance from origin within which objects are to be placed
        '''
        
        for i in range(n):
            if random.uniform(0,1) > 0.85:    
                box_size = self.box_arr[random.randint(0, len(self.box_arr)-1)][0:3]
                x = random.uniform(-max_length, max_length)  
                y = random.uniform(-max_length, max_length)
                self.create_box([x, y, 0.0], [random.uniform(-0.08, 0.08), random.uniform(-0.05, 0.05), random.uniform(-0.08, 0.08), 1], box_size)
            
            else:
                if random.randint(0,1):    
                    x = random.uniform(-max_length, max_length)  
                    y = random.uniform(-max_length, max_length)
                    z = random.uniform(-0.11, -0.08)
                    self.create_sphere([x, y, z], '14cm')
                else:
                    x = random.uniform(-max_length, max_length)  
                    y = random.uniform(-max_length, max_length)
                    z = random.uniform(-0.18, -0.15)
                    self.create_sphere([x, y, z], '20cm')