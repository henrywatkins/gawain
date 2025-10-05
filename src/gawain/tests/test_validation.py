"""Test script to demonstrate Pydantic validation for GAWAIN parameters"""

import numpy as np

from gawain.main import run_gawain


def test_valid_config():
    """Test with a valid configuration"""
    print("Testing valid configuration...")

    # Valid configuration
    nx, ny, nz = 64, 1, 1
    mesh_shape = (nx, ny, nz)
    lx, ly, lz = 1.0, 0.001, 0.001
    mesh_size = (lx, ly, lz)

    x = np.linspace(0.0, lx, num=nx)
    y = np.linspace(0.0, ly, num=ny)
    z = np.linspace(0.0, lz, num=nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Simple initial condition
    rho = np.ones(mesh_shape)
    pressure = np.ones(mesh_shape) * 0.1
    mx = np.zeros(mesh_shape)
    my = np.zeros(mesh_shape)
    mz = np.zeros(mesh_shape)
    e = pressure / (1.4 - 1) + 0.5 * (mx**2 + my**2 + mz**2) / rho
    initial_condition = np.array([rho, mx, my, mz, e])

    config = {
        "run_name": "validation_test",
        "cfl": 0.5,
        "mesh_shape": mesh_shape,
        "mesh_size": mesh_size,
        "mesh_grid": (X, Y, Z),
        "t_max": 0.001,  # Very short simulation
        "n_dumps": 1,
        "initial_condition": initial_condition,
        "boundary_type": ["periodic", "periodic", "periodic"],
        "adi_idx": 1.4,
        "integrator": "euler",
        "fluxer": "hll",
        "output_dir": "runs",
        "with_mhd": False,
    }

    try:
        run_gawain(config)
        print("✅ Valid configuration passed successfully!")
    except Exception as e:
        print(f"❌ Unexpected error with valid config: {e}")


def test_invalid_configs():
    """Test various invalid configurations to verify validation"""
    print("\nTesting invalid configurations...")

    # Base valid config
    nx, ny, nz = 32, 1, 1
    mesh_shape = (nx, ny, nz)
    lx, ly, lz = 1.0, 0.001, 0.001
    mesh_size = (lx, ly, lz)

    x = np.linspace(0.0, lx, num=nx)
    y = np.linspace(0.0, ly, num=ny)
    z = np.linspace(0.0, lz, num=nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    rho = np.ones(mesh_shape)
    pressure = np.ones(mesh_shape) * 0.1
    mx = np.zeros(mesh_shape)
    my = np.zeros(mesh_shape)
    mz = np.zeros(mesh_shape)
    e = pressure / (1.4 - 1) + 0.5 * (mx**2 + my**2 + mz**2) / rho
    initial_condition = np.array([rho, mx, my, mz, e])

    base_config = {
        "run_name": "test",
        "cfl": 0.5,
        "mesh_shape": mesh_shape,
        "mesh_size": mesh_size,
        "mesh_grid": (X, Y, Z),
        "t_max": 0.001,
        "n_dumps": 1,
        "initial_condition": initial_condition,
        "boundary_type": ["periodic", "periodic", "periodic"],
        "adi_idx": 1.4,
        "integrator": "euler",
        "fluxer": "hll",
        "output_dir": "runs",
        "with_mhd": False,
    }

    # Test cases for validation
    test_cases = [
        {
            "name": "Invalid CFL > 1",
            "config": {**base_config, "cfl": 1.5},
            "expected_error": "cfl",
        },
        {
            "name": "Negative CFL",
            "config": {**base_config, "cfl": -0.1},
            "expected_error": "cfl",
        },
        {
            "name": "Invalid integrator",
            "config": {**base_config, "integrator": "invalid_integrator"},
            "expected_error": "integrator",
        },
        {
            "name": "Invalid fluxer",
            "config": {**base_config, "fluxer": "invalid_fluxer"},
            "expected_error": "fluxer",
        },
        {
            "name": "Negative time",
            "config": {**base_config, "t_max": -1.0},
            "expected_error": "t_max",
        },
        {
            "name": "Invalid adiabatic index <= 1",
            "config": {**base_config, "adi_idx": 1.0},
            "expected_error": "adi_idx",
        },
        {
            "name": "Zero mesh dimension",
            "config": {**base_config, "mesh_shape": (0, 1, 1)},
            "expected_error": "mesh_shape",
        },
        {
            "name": "Negative mesh size",
            "config": {**base_config, "mesh_size": (-1.0, 0.001, 0.001)},
            "expected_error": "mesh_size",
        },
        {
            "name": "Wrong number of boundary conditions",
            "config": {**base_config, "boundary_type": ["periodic", "periodic"]},
            "expected_error": "boundary_type",
        },
        {
            "name": "Invalid boundary type",
            "config": {
                **base_config,
                "boundary_type": ["invalid", "periodic", "periodic"],
            },
            "expected_error": "boundary_type",
        },
    ]

    for test_case in test_cases:
        try:
            from gawain.io import Parameters

            params = Parameters(test_case["config"])
            print(f"❌ {test_case['name']}: Should have failed but didn't!")
        except ValueError as e:
            if test_case["expected_error"] in str(e).lower():
                print(f"✅ {test_case['name']}: Correctly caught validation error")
            else:
                print(
                    f"⚠️  {test_case['name']}: Caught error but not the expected one: {e}"
                )
        except Exception as e:
            print(f"⚠️  {test_case['name']}: Unexpected error type: {e}")


def test_physical_validation():
    """Test physical constraint validation"""
    print("\nTesting physical constraint validation...")

    nx, ny, nz = 32, 1, 1
    mesh_shape = (nx, ny, nz)
    lx, ly, lz = 1.0, 0.001, 0.001
    mesh_size = (lx, ly, lz)

    x = np.linspace(0.0, lx, num=nx)
    y = np.linspace(0.0, ly, num=ny)
    z = np.linspace(0.0, lz, num=nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    base_config = {
        "run_name": "test",
        "cfl": 0.5,
        "mesh_shape": mesh_shape,
        "mesh_size": mesh_size,
        "mesh_grid": (X, Y, Z),
        "t_max": 0.001,
        "n_dumps": 1,
        "boundary_type": ["periodic", "periodic", "periodic"],
        "adi_idx": 1.4,
        "integrator": "euler",
        "fluxer": "hll",
        "output_dir": "runs",
        "with_mhd": False,
    }

    # Test negative density
    rho_negative = np.ones(mesh_shape)
    rho_negative[0, 0, 0] = -1.0  # One negative density
    pressure = np.ones(mesh_shape) * 0.1
    mx = np.zeros(mesh_shape)
    my = np.zeros(mesh_shape)
    mz = np.zeros(mesh_shape)
    e = pressure / (1.4 - 1) + 0.5 * (mx**2 + my**2 + mz**2) / rho_negative
    initial_condition_negative_rho = np.array([rho_negative, mx, my, mz, e])

    try:
        from gawain.io import Parameters

        params = Parameters(
            {**base_config, "initial_condition": initial_condition_negative_rho}
        )
        print("❌ Negative density: Should have failed but didn't!")
    except ValueError as e:
        if "density" in str(e).lower():
            print("✅ Negative density: Correctly caught validation error")
        else:
            print(f"⚠️  Negative density: Unexpected error: {e}")

    # Test negative pressure (internal energy)
    rho = np.ones(mesh_shape)
    mx = np.ones(mesh_shape) * 10.0  # High momentum
    my = np.zeros(mesh_shape)
    mz = np.zeros(mesh_shape)
    e = np.ones(mesh_shape) * 0.01  # Very low total energy
    initial_condition_negative_pressure = np.array([rho, mx, my, mz, e])

    try:
        from gawain.io import Parameters

        params = Parameters(
            {**base_config, "initial_condition": initial_condition_negative_pressure}
        )
        print("❌ Negative pressure: Should have failed but didn't!")
    except ValueError as e:
        if "energy" in str(e).lower() or "pressure" in str(e).lower():
            print("✅ Negative pressure: Correctly caught validation error")
        else:
            print(f"⚠️  Negative pressure: Unexpected error: {e}")


if __name__ == "__main__":
    print("🧪 GAWAIN Parameter Validation Tests")
    print("=" * 50)

    test_valid_config()
    test_invalid_configs()
    test_physical_validation()

    print("\n" + "=" * 50)
    print("🎉 Validation testing complete!")
