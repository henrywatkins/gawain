"""MHD rotor test script"""

import numpy as np
from scipy.constants import pi as PI

from gawain.main import run_gawain

run_name = "3d_blast_wave"
output_dir = "runs"

cfl = 0.1
with_mhd = True

t_max = 0.01

integrator = "euler"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 64, 64, 64

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1, 1, 1

mesh_size = (lx, ly, lz)

x = np.linspace(-0.5, 0.5, num=nx)
y = np.linspace(-0.5, 0.5, num=ny)
z = np.linspace(-0.5, 0.5, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 1.4

R = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
R0 = 0.1

inner_mask = np.where(R <= R0)

rho = np.ones(mesh_shape)



mx = np.zeros(mesh_shape)
my = np.zeros(mesh_shape)
mz = np.zeros(mesh_shape)

bx = 100 * np.ones(mesh_shape) / np.sqrt(4 * PI)
by = np.zeros(mesh_shape)
bz = np.zeros(mesh_shape)

pressure = 0.1*np.ones(mesh_shape)
pressure[inner_mask] = 1000.0

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
