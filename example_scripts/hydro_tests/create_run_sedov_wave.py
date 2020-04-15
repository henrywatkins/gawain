''' test file for input '''

import numpy as np
import pickle

run_name = "sedov_blast_wave"

cfl = 0.01

with_gpu = False

t_max = 1.0

# "euler", "rk2", "leapfrog", "predictor-corrector"
integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 100, 150, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 1.5, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(-lx/2., lx/2.,num=nx)
y = np.linspace(-ly/2., ly/2.,num=ny)
z = np.linspace(0.0, lz,num=nz)
X,Y,Z =np.meshgrid(x,y,z, indexing='ij')

############ INITIAL CONDITION #################

adiabatic_idx = 5.0/3.0

rho = np.ones(X.shape)

R = np.sqrt(X**2+Y**2)

pressure = np.piecewise(R, [R < 0.1, R >= 0.1], [10.0, 0.1])

mx = np.zeros(X.shape)
my = mx
mz = mx

e = pressure/(adiabatic_idx-1.) + mx*mx/rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ['periodic', 'periodic', 'periodic']

############## DO NOT EDIT BELOW ############################
param_dict = {"run_name":run_name,"cfl":cfl, "mesh_shape":mesh_shape,
              "mesh_size":mesh_size, "t_max":t_max, "n_dumps":n_outputs,
              "using_gpu":with_gpu, "initial_con":initial_condition,
              "bound_cons":boundary_conditions, "adi_idx":adiabatic_idx,
              "integrator":integrator, "fluxer":fluxer}

param_filename = run_name+".pkl"

with open(param_filename, 'wb') as file:
    pickle.dump(param_dict, file)
