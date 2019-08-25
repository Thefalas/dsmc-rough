# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:23:14 2019

@author: malopez
"""
import numpy as np


def initialize_traslational_velocities(N, initial_temperature, theta):
    """ This function creates a numpy array of shape (N, 3) of velocities.
        The temperature (defined as T=2/3*v2) of said array is given by
        second argument. The velocity of the center of mass will be 0 """
    # T_r/T_t = theta; T = 0.5*(T_r + T_t); T=initial_temperature
    initial_temperature = 2 * initial_temperature / (1 + theta)
    # The following formula is correct (assuming that T= 2/3 * v2):
    # initial_temperature / std_normalDistribution = 2*std_normalDistribution
    # then --> std_normalDistribution = sqrt(initial_temperature/2)    
    std_normalDistribution = np.sqrt(initial_temperature/2)
    
    # Generating the array from a Gaussian distribution whith the above std
    # will give us the desired initial temperature
    tras_vels = np.random.normal(loc=0, scale=std_normalDistribution, size=(N, 3))

    # We now scale the velocity so that the velocity of the center of mass 
    # is initialized to 0. (Pöschel pag.203)
    tras_vels -= np.mean(tras_vels, axis=0)      
    return tras_vels


def initialize_rotational_velocities(N, initial_temperature, theta, moment_inertia):
    """ This function creates a numpy array of shape (N, 3) of velocities.
        The temperature (defined as T=2/3*v2) of said array is given by
        second argument. The velocity of the center of mass will be 0 """
    # T_r/T_t = theta; T = 0.5*(T_r + T_t); T=initial_temperature
    initial_temperature = 2 * theta * initial_temperature / (1 + theta)
    # Inspired in the method from initialize_traslational_velocities.
    std_normalDistribution = np.sqrt(initial_temperature/(2*moment_inertia))
    
    # Generating the array from a Gaussian distribution whith the above std
    # will give us the desired initial temperature
    rot_vels = np.random.normal(loc=0, scale=std_normalDistribution, size=(N, 3))

    # We now scale w so that the mean rotational speed is 0
    rot_vels -= np.mean(rot_vels, axis=0)      
    return rot_vels


def initialize_positions(N, part_radius, LX, LY, LZ):
    """ This function creates a numpy array of shape (N, 3) of positions.
        The particles are confined inside a cube of dimensions LX, LY, LZ """
    # The boundaries in which uniform numbers are generated are set so that
    # particles do not cut the walls
    pos_x = np.random.uniform(part_radius, LX-part_radius, (N, 1))
    pos_y = np.random.uniform(part_radius, LY-part_radius, (N, 1))
    pos_z = np.random.uniform(part_radius, LZ-part_radius, (N, 1))
    
    pos = np.concatenate((pos_x, pos_y, pos_z), axis=1)    
# =============================================================================
#     # Now we refine this array so that the center of mass is exactly where
#     # it should be (this method may cause some particles to fall outside walls)
#     deviation_from_central_point = [LX/2, LY/2, LZ/2] - np.mean(pos, axis=0)
#     pos += deviation_from_central_point
# =============================================================================
    return pos