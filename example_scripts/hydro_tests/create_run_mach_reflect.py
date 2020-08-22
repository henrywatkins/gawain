""" test file for input """

import numpy as np
import pickle

run_name = "mach_reflection"

cfl = 0.01

with_gpu = False
with_mhd = False
with_thermal_conductivity = False
with_resistivity = False

t_max = 0.04

# "euler", "rk2", "leapfrog", "predictor-corrector"
integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 3096, 1, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 7.0 / 5.0

rho = np.ones(X.shape)

pressure = np.piecewise(
    X,
    [X < lx / 10.0, (X >= lx / 10.0) & (X <= 9 * lx / 10.0), X > 9 * lx / 10.0],
    [1000.0, 0.01, 100.0],
)

mx = np.zeros(X.shape)
my = mx
mz = mx

e = pressure / (adiabatic_idx - 1) + mx * mx / rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed, reflective
boundary_conditions = ["reflective", "periodic", "periodic"]

############## DO NOT EDIT BELOW ############################
param_dict = {
    "run_name": run_name,
    "cfl": cfl,
    "mesh_shape": mesh_shape,
    "mesh_size": mesh_size,
    "t_max": t_max,
    "n_dumps": n_outputs,
    "using_gpu": with_gpu,
    "initial_con": initial_condition,
    "bound_cons": boundary_conditions,
    "adi_idx": adiabatic_idx,
    "integrator": integrator,
    "fluxer": fluxer,
}

param_filename = run_name + ".pkl"

with open(param_filename, "wb") as file:
    pickle.dump(param_dict, file)
