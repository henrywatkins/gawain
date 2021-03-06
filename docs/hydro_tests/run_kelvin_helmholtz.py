"""Kelvin-Helmholtz instability test script
"""

import numpy as np
from gawain.main import run_gawain

run_name = "kelvin_helmholtz"
output_dir = "."
cfl = 0.5

with_mhd = False

t_max = 5.0

# "euler",
integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 256, 256, 1

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

seedx = 0.02 * np.random.rand(*mesh_shape)
vy = 0.02 * np.random.rand(*mesh_shape)

pressure = 2.5 * np.ones(mesh_shape)

mx = rho * vx * (1.0 + seedx)
my = rho * vy
mz = np.zeros(mesh_shape)

e = pressure / (adiabatic_idx - 1.0) + 0.5 * (mx ** 2 + my ** 2 + mz ** 2) / rho

initial_condition = np.array([rho, mx, my, mz, e])

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
