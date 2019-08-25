# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:56:06 2019

@author: malopez
"""
import numpy as np


def calculate_traslational_temperature(tras_vels):
    T_tras = (2/3)*(np.linalg.norm(tras_vels, axis=1)**2).mean()
    return T_tras


def calculate_rotational_temperature(rot_vels, moment_inertia):
    T_rot = (2/3)*moment_inertia*(np.linalg.norm(rot_vels, axis=1)**2).mean()
    return T_rot


def calculate_temperature(tras_vels, rot_vels, moment_inertia):
    T_tras = calculate_traslational_temperature(tras_vels)
    T_rot = calculate_rotational_temperature(rot_vels, moment_inertia)
    T = 0.5*(T_tras + T_rot)
    return T