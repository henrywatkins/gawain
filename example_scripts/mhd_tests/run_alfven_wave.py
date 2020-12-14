"""Alfven wave MHD test script

This test aims to simulate a circularly-polarised
Alfven wave.
"""

import numpy as np
import scipy.constants as cnst
from gawain.main import run_gawain

run_name = "alfven_wave"
output_dir = "."

cfl = 0.5
with_mhd = True

t_max = 2.0

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 128, 128, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

alpha = cnst.pi / 6.0
sin = np.sin(alpha)
cos = np.cos(alpha)

lx, ly, lz = 1 / cos, 1 / sin, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 5 / 3

rho = np.ones(mesh_shape)

dr = 2 * cnst.pi * (X * cos + Y * sin)

mx = cos - 0.1 * np.sin(dr) * sin
my = sin + 0.1 * np.sin(dr) * cos
mz = rho * 0.1 * np.cos(dr)

bx = cos - 0.1 * np.sin(dr) * sin
by = sin + 0.1 * np.sin(dr) * cos
bz = 0.1 * np.cos(dr)

pressure = 0.1 * np.ones(mesh_shape)
mag_pressure = 0.5 * (bx * bx + by * by + bz * bz)

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
