__version__ = '0.1.0'

import numpy as np
import pandas as pd
import params
from initialization import initialize_positions, initialize_traslational_velocities, initialize_rotational_velocities
from stats import calculate_traslational_temperature, calculate_rotational_temperature, calculate_temperature
from propagation import propagate
from noise import apply_noise_pulse
from collision import compute_collisions


if __name__ == "__main__":

    # SEEDING THE GENERATOR
    try:
        # Using Intel's Math Kernel Library
        # https://software.intel.com/en-us/blogs/2016/06/15/faster-random-number-generation-in-intel-distribution-for-python
        # 'MCG31' gives the best performance, 'MT2203' provides better randomness
        # https://software.intel.com/en-us/mkl-vsnotes-basic-generators
        np.random_intel.seed(brng='MT2203')
    except:
        print('Intel libraries couldn\'t be used, falling back to standard Numpy')
        np.random.seed(params.seed)
    
    # SYSTEM INITIALIZATION
    tras_vels = initialize_traslational_velocities(params.N, params.initial_temperature, params.theta)
    rot_vels = initialize_rotational_velocities(params.N, params.initial_temperature, params.theta, params.inertia)
    
    pos = initialize_positions(params.N, params.effective_radius, params.LX, params.LY, params.LZ)    
    
    # INITIAL VALUES OF SOME STATISTICS
    T_tras =  calculate_traslational_temperature(tras_vels)
    T_rot = calculate_rotational_temperature(rot_vels, params.inertia)
    T = calculate_temperature(tras_vels, rot_vels, params.inertia)
    print('Traslational temperature:', T_tras)
    print('Rotational temperature:', T_rot )
    print('Total temperature:', T)
    
    # MAIN LOOP
    measurements = []
    cols_per_particle = 0
    rem = 0 
    for step in range(params.max_steps):       
        # First, we "estimate" grosso modo the maximum relative velocity 
        # between particles. This method will always overestimate but that is safer
        v_max = params.fwr * np.sqrt(T_tras)
        # Then we have to determine the maximum number of candidate collisions   
        n_cols_max = (params.N * v_max * params.dt / 2) + rem 
        # Remaining collisions (<1) (to be computed in next time_step)
        rem = n_cols_max - int(n_cols_max)
        # This way, we only use the integer part
        n_cols_max = int(n_cols_max)
        
        # With all this we can compute the results of the valid collisions
        tras_vels, n_valid_cols = compute_collisions(tras_vels, n_cols_max, v_max, params.alpha, params.dt)
        
        # Updating cols_per_particle value for this step
        cols_per_particle += (n_valid_cols / params.N)
        
# =============================================================================
#         # <Putting this outside the 'params.noise_interval' if loop is much slower
#         # but one is sure that the effect of the noise pulses is perfect>
#         # Finally we apply a random kick (white noise) to every particle
#         tras_vels = apply_noise_pulse(tras_vels, params.noise)
# =============================================================================
        
        # Every 'save_interval' steps we perform some computationally difficult
        # operations such as temperature measurement or 
        # data saving to an external file.
        if (step % params.save_interval) == 0:    
            time = step * params.dt
            T_tras = calculate_traslational_temperature(tras_vels)
            print(100*step/params.max_steps, '% \t Traslational temp:', T_tras)
            # Append current measurements, create and save DataFrame
            measurements.append([time, T_tras])
            data = pd.DataFrame(measurements, columns=['time', 'T_tras'])
            data.to_csv(params.output_file, sep='\t', index=False, header=False)
            

        # TODO: comprobar que cambiar noise_interval no afecta a la T_estacionaria
        # To save computation time the noise is only applied 
        # every 'noise_interval' steps
        if (step % params.noise_interval) == 0:
            # Finally we apply a random kick (white noise) to every particle
            # The noise has been 'charging' for a time: noise_interval*dt, so 
            # the longer the time since last kick, the stronger the next will be.
            # This intensity 'charging' increases with the sqrt of waiting time
            noise_intensity = np.sqrt(params.noise_interval*params.dt)*params.noise
            tras_vels = apply_noise_pulse(tras_vels, noise_intensity)
        
        pass