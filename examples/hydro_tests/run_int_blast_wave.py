"""2 interacting blast waves test script"""

import numpy as np

from gawain.main import run_gawain

# Woodward colella blast wave problem
run_name = "blast_wave"
output_dir = "."

cfl = 0.1

with_mhd = False

t_max = 0.04

# "euler", "rk2", "leapfrog", "predictor-corrector"
integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 128, 1, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 1.4

rho = np.ones(mesh_shape)

pressure = np.piecewise(
    X,
    [X < lx / 10.0, (X >= lx / 10.0) & (X <= 9 * lx / 10.0), X > 9 * lx / 10.0],
    [1000.0, 0.01, 100.0],
)

mx = np.zeros(mesh_shape)
my = np.zeros(mesh_shape)
mz = np.zeros(mesh_shape)

e = pressure / (adiabatic_idx - 1) + 0.5 * (mx**2 + my**2 + mz**2) / rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed, reflective
boundary_conditions = ["reflective", "periodic", "periodic"]

############## DO NOT EDIT BELOW ############################
config = {
    "run_name": run_name,
    "cfl": cfl,
    "mesh_shape": mesh_shape,
    "mesh_size": mesh_size,
    "t_max": t_max,
    "n_dumps": n_outputs,
    "initial_condition": initial_condition,
    "boundary_type": boundary_conditions,
    "adi_idx": adiabatic_idx,
    "integrator": integrator,
    "fluxer": fluxer,
    "output_dir": output_dir,
    "with_mhd": with_mhd,
}

run_gawain(config)
