# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gawain is a 2D/3D magnetohydrodynamic (MHD) simulation code written in Python. It supports:
- 2D/3D inviscid, compressible hydrodynamics
- 2D/3D ideal magnetohydrodynamics
- Boundary conditions: fixed-value, reflective, periodic, outflow
- Optional gravitational fields and arbitrary source functions
- CPU (NumPy) and GPU (CuPy) backends

## Architecture

### Core Data Flow

```
Configuration Dict → run_gawain() → Pydantic Validation → Parameters Object
    → SolutionVector Initialization → Timestep Loop → HDF5 Output
```

### Key Components

1. **`main.py`**: Single entry point
   - `run_gawain(config)`: Main function that orchestrates the entire simulation

2. **`config.py`**: Configuration validation via Pydantic
   - `SimulationConfig`: Validates all user inputs
   - Enums: `IntegratorType`, `FluxerType`, `BoundaryType`

3. **`io.py`**: I/O and parameter management
   - `Parameters`: Wraps validated config and creates simulation objects
   - `GawainData`: Reader class for HDF5 output files

4. **`numerics.py`**: State representation
   - `SolutionVector`: Hydro state (5 variables: density, mx, my, mz, energy)
   - `MHDSolutionVector`: MHD state (8 variables: adds bx, by, bz)
   - Shape: `(n_vars, nx, ny, nz)`

5. **`fluxes.py`**: Flux calculation methods
   - `FluxCalculator` (base class)
   - Subclasses: `LaxWendroffFlux`, `LaxFriedrichsFlux`, `HLLFlux`
   - Computes flux divergence for Euler/MHD equations

6. **`integrators.py`**: Time integration
   - `Integrator`: Orchestrates flux calculations, source terms, and gravity

7. **`backend.py`**: CPU/GPU abstraction
   - Provides `xp` variable (NumPy or CuPy based on `GAWAIN_USE_GPU` env var)
   - Functions: `to_cpu()`, `to_gpu()`, `synchronize()`

### Configuration Pattern

All simulations pass a dict to `run_gawain()`. Required fields:

```python
config = {
    "run_name": str,                    # Output file name
    "cfl": float,                       # 0 < cfl <= 1
    "mesh_shape": (nx, ny, nz),         # Grid dimensions
    "mesh_size": (lx, ly, lz),          # Physical domain size
    "mesh_grid": (X, Y, Z),             # From np.meshgrid(..., indexing="ij")
    "t_max": float,                     # Maximum simulation time
    "n_dumps": int,                     # Number of output snapshots
    "initial_condition": np.array,      # (5, nx, ny, nz) for hydro, (8,...) for MHD
    "boundary_type": [str, str, str],   # One per dimension (x, y, z)
    "adi_idx": float,                   # Adiabatic index (> 1.0)
    "integrator": str,                  # "euler" (only option currently)
    "fluxer": str,                      # "hll", "lax-wendroff", "lax-friedrichs", "base"
    "output_dir": str,                  # Directory for output files
    "with_mhd": bool,                   # Enable MHD equations
}
```

## Development Commands

Use uv for all development commands, including packaging, testing, and formatting.

### Installation

```bash
# Standard installation
pip install -e .

# With GPU support (NVIDIA CUDA)
pip install -e .[gpu]

# With development tools
pip install -e .[dev]

# AMD GPU (ROCm) - requires manual CuPy installation first
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm-6.4.1
export HCC_AMDGPU_TARGET=gfx942
pip install cupy
pip install -e .
```

### Testing

```bash
# Run all tests
pytest src/gawain/tests/

# Run specific test file
pytest src/gawain/tests/test_validation.py

# Run specific test function
pytest src/gawain/tests/test_validation.py::test_function_name

# Run with verbose output
pytest -v src/gawain/tests/
```

### Running Simulations

```bash
# From repository root
cd examples

# Hydro tests (Sod shock tube, etc.)
python hydro_tests/run_sod.py

# MHD tests (Brio-Wu shock tube, etc.)
python mhd_tests/run_brio_wu.py

# GPU mode (set environment variable)
GAWAIN_USE_GPU=1 python hydro_tests/run_sod.py
```

## Key Example Files

- Hydro example: `examples/hydro_tests/run_sod.py` (Sod shock tube)
- MHD example: `examples/mhd_tests/run_brio_wu.py` (Brio-Wu shock tube)
- Jupyter notebooks: `examples/example_gawain_notebook.ipynb`, `examples/example_plotting.ipynb`
- Validation tests: `src/gawain/tests/test_validation.py`
