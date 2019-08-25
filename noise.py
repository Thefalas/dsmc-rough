# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:11:59 2019

@author: malopez
"""
import numpy as np


def apply_noise_pulse(tras_vels, noise_intensity):
    """ Apply a random kick to every particle """
    tras_vels += np.random.normal(0, noise_intensity, tras_vels.shape)
    return tras_vels