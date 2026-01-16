"""
Generate validation figures for JOSS paper.

This script runs the Sod shock tube and Brio-Wu MHD shock tube tests
exactly as they appear in the examples directory, then generates plots
for the paper.

Run from project root: python paper/generate_paper_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gawain.main import run_gawain


def run_sod_shock_tube():
    """Run Sod shock tube test exactly as in examples/hydro_tests/run_sod.py"""
    print("Running Sod shock tube test...")

    run_name = "paper_sod"
    output_dir = "paper"

    cfl = 0.25
    with_mhd = False

    t_max = 0.25

    integrator = "euler"
    fluxer = "hll"

    # Mesh
    nx, ny, nz = 200, 1, 1
    mesh_shape = (nx, ny, nz)
    n_outputs = 100

    lx, ly, lz = 1.0, 0.001, 0.001
    mesh_size = (lx, ly, lz)

    x = np.linspace(0.0, lx, num=nx)
    y = np.linspace(0.0, ly, num=ny)
    z = np.linspace(0.0, lz, num=nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Initial condition
    adiabatic_idx = 7.0 / 5.0

    rho = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.125])
    pressure = np.piecewise(X, [X < 0.5, X >= 0.5], [1.0, 0.1])

    mx = np.zeros(X.shape)
    my = mx
    mz = mx

    e = pressure / (adiabatic_idx - 1) + 0.5 * mx * mx / rho

    initial_condition = np.array([rho, mx, my, mz, e])

    # Boundary condition
    boundary_conditions = ["fixed", "periodic", "periodic"]

    # Configuration
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

    # Load results
    with h5py.File("paper/paper_sod.h5", "r") as f:
        solution = f["solutions"][-1]  # Final timestep
        x_grid = f["X"][:, 0, 0]

    return x_grid, solution, adiabatic_idx


def run_brio_wu_tube():
    """Run Brio-Wu MHD shock tube test exactly as in examples/mhd_tests/run_brio_wu.py"""
    print("Running Brio-Wu MHD shock tube test...")

    run_name = "paper_briowu"
    output_dir = "paper"

    cfl = 0.8
    with_mhd = True

    t_max = 0.25

    integrator = "euler"
    fluxer = "hll"

    # Mesh
    nx, ny, nz = 800, 1, 1
    mesh_shape = (nx, ny, nz)
    n_outputs = 100

    lx, ly, lz = 1.0, 0.001, 0.001
    mesh_size = (lx, ly, lz)

    x = np.linspace(0.0, lx, num=nx)
    y = np.linspace(0.0, ly, num=ny)
    z = np.linspace(0.0, lz, num=nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Initial condition
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

    # Boundary condition
    boundary_conditions = ["fixed", "periodic", "periodic"]

    # Configuration
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

    # Load results
    with h5py.File("paper/paper_briowu.h5", "r") as f:
        solution = f["solutions"][-1]  # Final timestep
        x_grid = f["X"][:, 0, 0]

    return x_grid, solution, adiabatic_idx


def generate_figures():
    """Generate all figures for the paper."""
    print("\n" + "="*60)
    print("GENERATING FIGURES FOR JOSS PAPER")
    print("="*60)

    # Create output directory
    os.makedirs("paper/figures", exist_ok=True)

    # Run Sod shock tube
    x_sod, sol_sod, gamma_sod = run_sod_shock_tube()

    # Extract numerical solution
    rho_sod = sol_sod[0, :, 0, 0]
    mx_sod = sol_sod[1, :, 0, 0]
    u_sod = mx_sod / rho_sod
    e_sod = sol_sod[4, :, 0, 0]
    p_sod = (gamma_sod - 1) * (e_sod - 0.5 * mx_sod**2 / rho_sod)

    print("\nGenerating Sod shock tube figure...")

    # Figure 1: Sod shock tube
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x_sod, rho_sod, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Sod Shock Tube: Density')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_sod, u_sod, 'b-', linewidth=1.5)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Sod Shock Tube: Velocity')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_sod, p_sod, 'b-', linewidth=1.5)
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Pressure')
    axes[2].set_title('Sod Shock Tube: Pressure')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/figures/sod_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/sod_validation.pdf', bbox_inches='tight')
    print("Saved: paper/figures/sod_validation.png")
    plt.close()

    # Run Brio-Wu
    x_bw, sol_bw, gamma_bw = run_brio_wu_tube()

    # Extract MHD solution
    rho_bw = sol_bw[0, :, 0, 0]
    mx_bw = sol_bw[1, :, 0, 0]
    u_bw = mx_bw / rho_bw
    e_bw = sol_bw[4, :, 0, 0]
    bx_bw = sol_bw[5, :, 0, 0]
    by_bw = sol_bw[6, :, 0, 0]
    bz_bw = sol_bw[7, :, 0, 0]
    mag_pressure_bw = 0.5 * (bx_bw**2 + by_bw**2 + bz_bw**2)
    p_bw = (gamma_bw - 1) * (e_bw - 0.5 * mx_bw**2 / rho_bw - mag_pressure_bw)

    print("\nGenerating Brio-Wu MHD shock tube figure...")

    # Figure 2: Brio-Wu MHD shock tube
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x_bw, rho_bw, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Brio-Wu MHD: Density')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_bw, by_bw, 'b-', linewidth=1.5)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('$B_y$')
    axes[1].set_title('Brio-Wu MHD: Transverse Magnetic Field')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_bw, p_bw, 'b-', linewidth=1.5)
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Pressure')
    axes[2].set_title('Brio-Wu MHD: Pressure')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/figures/briowu_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/briowu_validation.pdf', bbox_inches='tight')
    print("Saved: paper/figures/briowu_validation.png")
    plt.close()

    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - paper/figures/sod_validation.png (and .pdf)")
    print("  - paper/figures/briowu_validation.png (and .pdf)")


if __name__ == "__main__":
    generate_figures()
