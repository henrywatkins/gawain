''' test file for input '''

import numpy as np
import pickle

run_name = "sod_shock_tube"

cfl = 0.01

with_gpu = False

t_max = 0.25

# "euler", "rk2", "leapfrog", "predictor-corrector"
integrator = "leapfrog"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "lax-wendroff"

################ MESH #####################

nx, ny, nz = 100, 1, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx,num=nx)
y = np.linspace(0.0, ly,num=ny)
z = np.linspace(0.0, lz,num=nz)
X,Y,Z =np.meshgrid(x,y,z, indexing='ij')

############ INITIAL CONDITION #################

adiabatic_idx = 7.0/5.0

rho = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.125])

pressure = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.1])

mx = np.zeros(X.shape)
my = mx
mz = mx

e = pressure/(adiabatic_idx-1) + mx*mx/rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ['fixed', 'periodic', 'periodic']

############## DO NOT EDIT BELOW ############################
param_dict = {"run_name":run_name,"cfl":cfl, "mesh_shape":mesh_shape,
              "mesh_size":mesh_size, "t_max":t_max, "n_dumps":n_outputs,
              "using_gpu":with_gpu, "initial_con":initial_condition,
              "bound_cons":boundary_conditions, "adi_idx":adiabatic_idx,
              "integrator":integrator, "fluxer":fluxer}

param_filename = run_name+".pkl"

with open(param_filename, 'wb') as file:
    pickle.dump(param_dict, file)
