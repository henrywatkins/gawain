"""Rayleigh-Taylor instability script

This script runs the hydrodynamic Rayleigh-Taylor
instability test
"""

import numpy as np
from gawain.main import run_gawain
from scipy.constants import pi as PI

run_name = "rayleigh_taylor"
output_dir = "."
cfl = 0.5

with_mhd = False

t_max = 8.0

# "euler",
integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 100, 300, 1

mesh_shape = (nx, ny, nz)

n_outputs = 200

lx, ly, lz = 0.5, 1.5, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(-lx / 2.0, lx / 2.0, num=nx)
y = np.linspace(-ly / 2.0, ly / 2.0, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 1.4

rho = np.piecewise(Y, [Y > 0, Y <= 0], [2.0, 1.0])

g = 0.1

gravity_field = np.array(
    [np.zeros(mesh_shape), g * np.ones(mesh_shape), np.zeros(mesh_shape)]
)

P0 = 2.5 * np.ones(mesh_shape)

pressure = P0 - g * rho * Y

mx = np.zeros(mesh_shape)
my = rho * 0.01 * 0.25 * (1.0 + np.cos(4 * PI * X)) * (1.0 + np.cos(3 * PI * Y))
mz = np.zeros(mesh_shape)

e = pressure / (adiabatic_idx - 1.0) + 0.5 * (mx ** 2 + my ** 2 + mz ** 2) / rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ["periodic", "reflective", "periodic"]

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
    "gravity": gravity_field,
    "with_mhd": with_mhd,
}

run_gawain(config)
