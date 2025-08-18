"""MHD rotor test script"""

import numpy as np
from scipy.constants import pi as PI

from gawain.main import run_gawain

run_name = "mhd_rotor"
output_dir = "runs"

cfl = 0.8
with_mhd = True

t_max = 0.15

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 256, 256, 1

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

# Parameters
r0 = 0.1
r1 = 0.115
u0 = 2.0

# Centered coordinates
xc, yc = 0.5, 0.5
dx = X - xc
dy = Y - yc
r = np.sqrt(dx**2 + dy**2)

# Transition function f(r)
f = np.zeros_like(r)
mask = (r > r0) & (r < r1)
f[mask] = (r1 - r[mask]) / (r1 - r0)
f[r <= r0] = 1.0
f[r >= r1] = 0.0

# Density
rho = np.ones_like(r)
rho[r <= r0] = 10.0
rho[mask] = 1.0 + 9.0 * f[mask]

# Velocities
u = np.zeros_like(r)
v = np.zeros_like(r)

# Inner rotor
mask_inner = r <= r0
u[mask_inner] = -f[mask_inner] * u0 * dy[mask_inner] / r0
v[mask_inner] = f[mask_inner] * u0 * dx[mask_inner] / r0

# Transition region
u[mask] = -f[mask] * u0 * dy[mask] / r[mask]
v[mask] = f[mask] * u0 * dx[mask] / r[mask]

# Pressure and B fields
pressure = np.ones_like(r)
bx = np.full_like(r, 5.0 / np.sqrt(4.0 * np.pi))
by = np.zeros_like(r)
bz = np.zeros_like(r)
w = np.zeros_like(r)

mx = rho * u
my = rho * v
mz = rho * w

mag_pressure = 0.5 * (bx**2 + by**2 + bz**2)

e = (
    pressure / (adiabatic_idx - 1)
    + 0.5 * (mx * mx + my * my + mz * mz) / rho
    + mag_pressure
)


initial_condition = np.array([rho, mx, my, mz, e, bx, by, bz])

import matplotlib.pyplot as plt

plt.imshow(rho[:, :, 0], cmap="viridis", origin="lower")
plt.colorbar()
plt.title("Density")
plt.savefig("density_initial_condition.png")

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
