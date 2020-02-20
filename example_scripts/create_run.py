''' test file for input '''

import numpy as np
import pickle

run_name = "input_test_run"

cfl = 0.25

nx, ny, nz = 100, 1, 1

mesh_shape = (nx, ny, nz)

t_max = 1.0

with_gpu = False

boundary_conditions = ['periodic', 'periodic', 'periodic']

n_outputs = 100

lx, ly, lz = 1.0, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx,num=nx)
y = np.linspace(0.0, ly,num=ny)
X,Y =np.meshgrid(x,y)

mu = 0.5
sigma = 0.1

initial_condition = np.exp(-0.5*((x-mu)/sigma)**2) 
#initial_condition = np.exp(-0.5*((X-mu)/sigma)**2-0.5*((Y-mu)/sigma)**2) 

############## DO NOT EDIT BELOW ############################
param_dict = {"run_name":run_name,"clf":cfl, "mesh_shape":mesh_shape, 
              "mesh_size":mesh_size, "t_max":t_max, "n_dumps":n_outputs, 
              "using_gpu":False, "initial_con":initial_condition, "bound_cons":boundary_conditions}

param_filename = run_name+".pkl"
              
with open(param_filename, 'wb') as file:
    pickle.dump(param_dict, file)

