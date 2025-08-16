"""Linear wave MHD test script

This test aims to simulate a linear wave.
"""

import numpy as np
import scipy.constants as cnst

from gawain.main import run_gawain

run_name = "linear_wave"
output_dir = "runs"

cfl = 0.5
with_mhd = True

t_max = 2.0

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 100, 1, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 5 / 3

rho = np.ones(mesh_shape)

mx = np.zeros(mesh_shape)
my = np.zeros(mesh_shape)
mz = np.zeros(mesh_shape)

bx = np.ones(mesh_shape) / np.sqrt(4 * cnst.pi)
by = np.sqrt(2) * np.ones(mesh_shape) / np.sqrt(4 * cnst.pi)
bz = 0.5 * np.ones(mesh_shape) / np.sqrt(4 * cnst.pi)

pressure = np.ones(mesh_shape) / adiabatic_idx + 0.5 * (bx * bx + by * by + bz * bz)

e = pressure / (adiabatic_idx - 1) + 0.5 * (mx * mx + my * my + mz * mz) / rho

constant = np.array([rho, mx, my, mz, e, bx, by, bz])

Ra = np.array([0.00, 0.00, -3.33e-1, 9.43e-1, 0.00, 0.00, -3.33e-1, 9.43e-1])
Rfms = np.array([4.47e-1, -8.94e-1, 4.21e-1, 1.49e-1, 2.01, 0.00, 8.43e-1, 2.98e-1])
Rsms = np.array(
    [8.94e-1, -4.47e-1, -8.43e-1, -2.98e-1, 6.7e-1, 0.00, -4.21e-1, -1.49e-1]
)

perturbation = 1e-4 * np.sin(2 * cnst.pi * X)
dU = np.array([d * perturbation for d in Rfms])

initial_condition = constant + dU

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ["periodic", "periodic", "periodic"]

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
