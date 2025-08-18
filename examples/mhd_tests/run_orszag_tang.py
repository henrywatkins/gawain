"""Orszag-Tang vortex MHD test script"""

import numpy as np
from scipy.constants import pi as PI

from gawain.main import run_gawain

run_name = "orszag_tang"
output_dir = "runs"

cfl = 0.25
with_mhd = True

t_max = 0.5

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 64, 64, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1, 1, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 5.0 / 3.0

rho = np.ones(mesh_shape)

mx = -np.sin(2 * PI * Y)
my = np.sin(2 * PI * X)
mz = np.zeros(mesh_shape)

B0 = 1 / adiabatic_idx

bx = -B0 * np.sin(2 * PI * Y)
by = B0 * np.sin(4 * PI * X)
bz = np.zeros(mesh_shape)

pressure = np.ones(mesh_shape) / adiabatic_idx
mag_pressure = 0.5 * (bx**2 + by**2 + bz**2)

e = (
    pressure / (adiabatic_idx - 1)
    + 0.5 * (mx * mx + my * my + mz * mz) / rho
    + mag_pressure
)

initial_condition = np.array([rho, mx, my, mz, e, bx, by, bz])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ["periodic", "periodic", "periodic"]

############## DO NOT EDIT BELOW ############################
config = {
    "run_name": run_name,
    "cfl": cfl,
    "mesh_shape": mesh_shape,
    "mesh_size": mesh_size,
    "mesh_grid": (X, Y, Z),
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
