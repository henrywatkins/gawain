# Gawain

Gawain is a 2D/3D magnetohydrodynamic simulation code written in python. This software currently supports simulations of the following physical systems:

- 2D/3D inviscid, compressible hydrodynamics
- 2D/3D ideal magnetohydrodynamics

Gawain currently supports the following boundary conditions:

- fixed - value
- reflective
- periodic
- outflow

Gawain can also be used to simulate hydrodynamics or magnetohydrodynamics in the presence of a gravitational field or an arbitrary source function.  


## Installation

Clone the repository and pip install 

```bash
pip install -e .
```


## Examples

A few example simulation scripts and jupyter notebooks can be found in the 'examples' directory. The notebooks provide a detailed how-to on creating and running a simulation. To run a simulation, one can either use a jupyter notebook or a python script. The simulation parameters are be created, grouped into a python dict, and passed to the `run_gawain` function. See an example below, or check out the examples directory for more test cases

 ```python
import numpy as np

from gawain.main import run_gawain

run_name = "sod_shock_tube"
output_dir = "runs"

cfl = 0.25
with_mhd = False

t_max = 0.25

integrator = "euler"
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
 ```

## Setting up a problem

To set up a problem, you need to define the mesh, initial conditions, boundary conditions, and simulation parameters. The mesh is defined by its shape and size, while the initial conditions are provided as a numpy array containing the relevant physical quantities (e.g., density, momentum components, energy). Boundary conditions can be specified as a list of strings corresponding to each spatial dimension. You also need to set the simulation parameters such as the CFL number, maximum simulation time, number of output dumps, adiabatic index, integrator type, and flux calculation method. All of these settings are then grouped into a configuration dictionary that is passed to the `run_gawain` function to start the simulation.

Options for integrators only includes "euler" at present, while flux calculation methods include "hll" (HLLF with minmod limiter), "lax-wendroff" and "lax-friedrichs". Boundary conditions can be "fixed", "reflective", "periodic" or "outflow".

# GPU Support

GAWAIN now supports GPU acceleration through CuPy, which provides a NumPy-compatible interface for GPU computing. The implementation uses a backend abstraction layer that allows seamless switching between CPU (NumPy) and GPU (CuPy) execution without code changes.

## Installation

To install GAWAIN with GPU support, use the following command:

```bash
pip install -e .[gpu]
```
and set the environment variable `GAWAIN_USE_GPU=1` before running your simulations. If a valid GPU setup is detected, GAWAIN will automatically utilize the GPU for computations.

## AMD GPU Support

AMD GPUs are supported through the ROCm platform. Ensure that you have ROCm installed and configured correctly on your system. To install CuPy with ROCm support, set the environment variables and install via a pip install, then install GAWAIN with as standard support:

```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm-6.4.1
export HCC_AMDGPU_TARGET=gfx942
pip install cupy
pip install -e .
```

You can check cupy has installed successfully by running:

```bash
python -c "import cupy; cupy.show_config()"
```

# Visualizing results

Output from the simulations are stored in hdf5 format. You can use the `h5py` package to read the output files. A simple visualization script using `matplotlib` is provided in the `examples` directory. Here you can find examples of the Reader and plotting utilities provided in the `gawain.io` module.