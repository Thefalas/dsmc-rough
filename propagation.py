# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:04:06 2019

@author: malopez
"""
import numpy as np


def propagate(t, pos, vel, LX, LY, LZ):
    
    # Free stream of particles
    pos += t*vel
    
    # This is to account for the periodic boundaries
    pos[:,0] -= np.floor(pos[:,0]/LX)*LX
    pos[:,1] -= np.floor(pos[:,1]/LY)*LY
    pos[:,2] -= np.floor(pos[:,2]/LZ)*LZ
    
    return pos