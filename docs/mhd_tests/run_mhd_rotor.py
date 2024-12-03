"""MHD rotor test script
"""

import numpy as np
from scipy.constants import pi as PI
from gawain.main import run_gawain

run_name = "mhd_rotor"
output_dir = "."

cfl = 0.25
with_mhd = True

t_max = 0.15

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 128, 128, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1, 1, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 1.4

R = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
R0 = 0.1
R1 = 0.115
FR = (R1 - R) / (R - R0)

U0 = 2


rho_mid_vals = 1 + 9 * FR
vx_in_vals = -FR * U0 * (Y - 0.5) / R0
vx_mid_vals = -FR * U0 * (Y - 0.5) / R
vy_in_vals = FR * U0 * (X - 0.5) / R0
vy_mid_vals = FR * U0 * (X - 0.5) / R

inner_mask = np.where(R <= R0)
middle_mask = np.where(np.logical_and(R > R0, R < R1))

rho = np.ones(mesh_shape)
rho[inner_mask] = 10.0
rho[middle_mask] = rho_mid_vals[middle_mask]

vx = np.zeros(mesh_shape)
vx[inner_mask] = vx_in_vals[inner_mask]
vx[middle_mask] = vx_mid_vals[middle_mask]

vy = np.zeros(mesh_shape)
vy[inner_mask] = vy_in_vals[inner_mask]
vy[middle_mask] = vy_mid_vals[middle_mask]

vz = np.zeros(mesh_shape)

mx = rho * vx
my = rho * vy
mz = rho * vz

bx = 5 * np.ones(mesh_shape) / np.sqrt(4 * PI)
by = np.zeros(mesh_shape)
bz = np.zeros(mesh_shape)

pressure = np.ones(mesh_shape)

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
