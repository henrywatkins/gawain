# Gawain

Gawain is a 2D magnetohydrodynamic simulation code written in python. This software currently supports simulations of the following physical systems:

- 2D inviscid, compressible hydrodynamics
- 2D ideal magnetohydrodynamics

Gawain currently supports the following boundary conditions:

- fixed - value
- reflective
- periodic
- outflow

Gawain can also be used to simulate hydrodynamics or magnetohydrodynamics in the presence of a gravitational field or an arbitrary source function.  

## Getting Started

#### Prerequisites

The aim of this code was simplicity, so the dependencies have been kept to a minimum. However, a few key external libraries are required and can be installed using 

```
 pip install -r requirements.txt
```

#### Installing

The package is set up using

```
python ./setup.py install
```

## Examples

A few example simulation scripts and jupyter notebooks can be found in the 'docs' directory. The notebooks provide a detailed how-to on creating and running a simulation.

To run a simulation, one can either use a jupyter notebook or a python script. The simulation parameters are be created, grouped into a python dict, and passed to the `run_gawain` function. See an example below, or check out the docs directory for more examples

 ```python
import numpy as np
from gawain.main import run_gawain

run_name = "sod_shock_tube"
output_dir = "."

cfl = 0.5
with_mhd = False
with_thermal_conductivity = False
with_resistivity = False

t_max = 0.25

integrator = "euler"
# "base", "lax-wendroff", "lax-friedrichs", "vanleer", "hll"
fluxer = "hll"

################ MESH #####################

nx, ny, nz = 200, 1, 1

mesh_shape = (nx, ny, nz)

n_outputs = 100

lx, ly, lz = 1.0, 0.001, 0.001

mesh_size = (lx, ly, lz)

x = np.linspace(0.0, lx, num=nx)
y = np.linspace(0.0, ly, num=ny)
z = np.linspace(0.0, lz, num=nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

############ INITIAL CONDITION #################

adiabatic_idx = 7.0 / 5.0

rho = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.125])

pressure = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.1])

mx = np.zeros(X.shape)
my = mx
mz = mx

e = pressure / (adiabatic_idx - 1) + 0.5 * mx * mx / rho

initial_condition = np.array([rho, mx, my, mz, e])

############## BOUNDARY CONDITION ######################
# available types: periodic, fixed, outflow, reflective
boundary_conditions = ["fixed", "periodic", "periodic"]

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
 ```

