import numpy as np

# ========= Simulation Variables =========
N = 50000 # Total number of particles in the system
dt = 0.001 # Time step duration
max_steps = 100000 # Number of steps
n_runs = 5 # Number of runs for each alpha
fwr = 6 # Parameter used for calculating maximum relative velocity
initial_temperature = 5 # Used to initialize velocity
noise = 0.265 # Used to regulate noise intensity level
noise_interval = 50 # Steps between noise pulses (doesn't affect overall noise intensity)
theta = 0.10 # Initial ratio T_rot/T_tras
seed = 20190725 # Seed for random number generator

# ========= Colisional properties =========
alpha = 0.85 # Normal restitution coefficient
beta = -0.5 # Tangetial restitution coefficient

# ========= System properties =========
LX = 200
LY = 200
LZ = 200
V = LX*LY*LZ

# ========= Particle properties =========
m = 1 # Mass
effective_diameter = 1
effective_radius = effective_diameter/2
particle_volume = (4/3)*np.pi*(effective_radius**3) # Volume occuppied by one particle
inertia = 2/5 # Moment of inertia for spherical particles

# ========= Useful statistics =========
packing_fraction = N*particle_volume/V # Particle density (packing fraction)
mean_free_path = 1 / (np.sqrt(2)*np.pi*(effective_diameter**2)*packing_fraction)
knudsen_number = mean_free_path / min(LX, LY, LZ)

# ========= Output settings =========
save_interval = 100 # Number of steps between consecutive saving of data
output_file = 'DSMC_beta'+str(beta)+'_alpha'+str(alpha)+'.data.dat'

# Bins, not used for the moment
n_bins = 50
bin_size = LZ/n_bins
ratio_bin_mfp = bin_size/mean_free_path