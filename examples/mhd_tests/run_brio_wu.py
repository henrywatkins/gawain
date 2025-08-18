"""test file for input"""

import numpy as np

from gawain.main import run_gawain

run_name = "brio_wu_tube"
output_dir = "runs"

cfl = 0.8
with_mhd = True
with_thermal_conductivity = False
with_resistivity = False

t_max = 0.25

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 800, 1, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 2.0

rho = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.125])

pressure = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.1])

mx = np.zeros(X.shape)
my = mx
mz = mx


bx = 0.75 * np.ones_like(X)
by = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, -1.0])
bz = np.zeros(X.shape)

mag_pressure = 0.5 * (bx**2 + by**2 + bz**2)

e = (
    pressure / (adiabatic_idx - 1)
    + 0.5 * (mx * mx + my * my + mz * mz) / rho
    + mag_pressure
)

initial_condition = np.array([rho, mx, my, mz, e, bx, by, bz])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed
boundary_conditions = ["fixed", "periodic", "periodic"]

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
