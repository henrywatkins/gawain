""" test file for input """

import numpy as np
from gawain.main import run_gawain

run_name = "kelvin_helmholtz"
output_dir = "."
cfl = 0.1

with_gpu = False

t_max = 5.0

# "euler", "rk2", "leapfrog", "predictor-corrector"
integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 200, 200, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 1.0, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(-lx / 2.0, lx / 2.0, num=nx)
y = np.linspace(-ly / 2.0, ly / 2.0, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 1.4

rho = np.piecewise(Y, [np.absolute(Y) > 0.25, np.absolute(Y) <= 0.25], [1.0, 2.0])
vx = np.piecewise(Y, [np.absolute(Y) > 0.25, np.absolute(Y) <= 0.25], [-0.5, 0.5])
vy = np.zeros(X.shape)
a = -0.1
b = 0.1

seed1 = a + (b - a) * np.random.random(X.shape)
seed2 = a + (b - a) * np.random.random(X.shape)

pressure = 2.5 * np.ones(X.shape)

mx = rho * vx * (1.0 + seed1)
my = rho * vy * (1.0 + seed2)
mz = np.zeros(X.shape)

e = pressure / (adiabatic_idx - 1.0) + mx * mx / rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ["periodic", "periodic", "periodic"]

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
    "output_dir": output_dir,
}

run_gawain(param_dict)
