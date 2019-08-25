# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:46:07 2019

@author: malopez
"""
import numpy as np


def compute_collisions(tras_vels, n_cols_max, v_max, alpha, dt):
    
    
    N = tras_vels.shape[0]

    # It is more efficient to generate all random numbers at once
    # We choose multiple (n_cols_max) random pairs of particles
    random_pairs = np.random.choice(N, size=(n_cols_max,2))
    # List of random numbers to use as collision criteria
    random_numbers = np.random.uniform(0,1, n_cols_max)

    # Now, we generate random directions, (modulus 1) sigmas
    costheta = np.random.uniform(0,2, size=n_cols_max) - 1
    sintheta = np.sqrt(1-costheta**2)
    phis = np.random.uniform(0,2*np.pi, size=n_cols_max)
    
    x_coord = sintheta*np.cos(phis)
    y_coord = sintheta*np.sin(phis)
    z_coord = costheta
    sigmas = np.stack((x_coord, y_coord, z_coord), axis=1)   
    
    # This is a vectorized method, it should be faster than the for loop
    # Using those random pairs we calculate relative velocities
    rel_vs = np.array(list(map(lambda i, j: tras_vels[i]-tras_vels[j], random_pairs[:,0], random_pairs[:,1])))
    # And now its modulus by performing a dot product with sigmas array
    rel_vs_mod = np.sum(rel_vs*sigmas, axis=1)
    
    # With this information we can check which collisions are valid
    ratios = rel_vs_mod / v_max
    valid_cols = ratios > random_numbers
    
    # The valid pairs of particles of each valid collision are:
    valid_pairs = random_pairs[valid_cols]
    
    # Number of collisions that take place in this step
    n_valid_cols = len(valid_pairs)   

    # Now, we select only those rows that correspond to valid collisions
    valid_dotProducts = rel_vs_mod[valid_cols]
    # See: https://stackoverflow.com/questions/16229823/how-to-multiply-numpy-2d-array-with-numpy-1d-array
    valid_vectors = sigmas[valid_cols] * valid_dotProducts[:, None]
    new_vel_components = 0.5*(1+alpha) * valid_vectors
    
    valid_is = valid_pairs[:,0]
    valid_js = valid_pairs[:,1]
    
    # Updating the velocities array with its new values
    tras_vels[valid_is] -= new_vel_components
    tras_vels[valid_js] += new_vel_components

    return tras_vels, n_valid_cols