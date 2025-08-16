"""Current sheet MHD test script"""

import numpy as np
from scipy.constants import pi as PI

from gawain.main import run_gawain

run_name = "current_sheet"
output_dir = "runs"

cfl = 0.25
with_mhd = True

t_max = 5.0

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 128, 128, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1, 1, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(-lx / 2, lx / 2, num=nx)
y = np.linspace(-ly / 2, ly / 2, num=ny)
z = np.linspace(0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 5 / 3

rho = np.ones(mesh_shape)
vx = 0.1 * np.sin(2 * PI * Y)
vy = np.zeros(mesh_shape)
vz = np.zeros(mesh_shape)

bx = np.zeros(mesh_shape)
by = np.piecewise(X, [np.absolute(X) < 0.25, np.absolute(X) >= 0.25], [1.0, -1.0])
bz = np.zeros(mesh_shape)

mx = rho * vx
my = rho * vy
mz = rho * vz

pressure = 0.3 * np.ones(mesh_shape)
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
