import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from gawain.fluxes import (FluxCalculator, HLLFluxer, LaxFriedrichsFluxer,
                           LaxWendroffFluxer)
from gawain.integrators import Integrator
from gawain.io import Output, Parameters, Reader
from gawain.numerics import MHDSolutionVector, SolutionVector


class MockFileHandle:
    """Mock file handle for testing file operations"""

    def __init__(self):
        self.content = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, content):
        self.content += content


class MockPatch:
    """Simple mock replacement for patch decorator"""

    def __init__(self, target, return_value=None, side_effect=None):
        self.target = target
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        self.call_args_list = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args_list.append((args, kwargs))
        if self.side_effect:
            return self.side_effect(*args, **kwargs)
        return self.return_value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def assert_called_once(self):
        assert self.call_count == 1

    def assert_called_once_with(self, *args, **kwargs):
        assert self.call_count == 1
        assert self.call_args_list[0] == (args, kwargs)

    def assert_not_called(self):
        assert self.call_count == 0


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing"""
    # Create mesh grids
    x = np.linspace(0, 1.0, 10)
    y = np.linspace(0, 0.8, 8)
    z = np.linspace(0, 0.6, 6)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    config = {
        "run_name": "test_run",
        "cfl": 0.5,
        "mesh_shape": (10, 8, 6),
        "mesh_size": (1.0, 0.8, 0.6),
        "mesh_grid": (X, Y, Z),
        "t_max": 1.0,
        "n_dumps": 5,
        "adi_idx": 1.4,
        "initial_condition": np.ones((5, 10, 8, 6)),
        "boundary_type": ["periodic", "periodic", "periodic"],
        "output_dir": ".",
        "with_mhd": False,
        "integrator": "euler",
        "fluxer": "hll",
    }
    return config


@pytest.fixture
def sample_mhd_config():
    """Sample MHD configuration dictionary for testing"""
    # Create mesh grids
    x = np.linspace(0, 1.0, 10)
    y = np.linspace(0, 0.8, 8)
    z = np.linspace(0, 0.6, 6)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    config = {
        "run_name": "test_mhd_run",
        "cfl": 0.5,
        "mesh_shape": (10, 8, 6),
        "mesh_size": (1.0, 0.8, 0.6),
        "mesh_grid": (X, Y, Z),
        "t_max": 1.0,
        "n_dumps": 5,
        "adi_idx": 1.4,
        "initial_condition": np.ones((8, 10, 8, 6)),
        "boundary_type": ["periodic", "periodic", "periodic"],
        "output_dir": ".",
        "with_mhd": True,
        "integrator": "euler",
        "fluxer": "hll",
    }
    return config


@pytest.fixture
def sample_solution_vector():
    """Create a sample solution vector for testing"""
    sv = SolutionVector()
    data = np.random.rand(5, 10, 8, 6)
    sv.data = data
    return sv


@pytest.fixture
def sample_mhd_solution_vector():
    """Create a sample MHD solution vector for testing"""
    mhd_sv = MHDSolutionVector()
    data = np.random.rand(8, 10, 8, 6)
    mhd_sv.data = data
    return mhd_sv


class TestParameters:
    def test_parameters_initialization(self, sample_config):
        """Test Parameters initialization"""
        params = Parameters(sample_config)

        assert params.run_name == "test_run"
        assert params.cfl == 0.5
        assert params.mesh_shape == (10, 8, 6)
        assert params.mesh_size == (1.0, 0.8, 0.6)
        assert params.t_max == 1.0
        assert params.n_outputs == 5
        assert params.adi_idx == 1.4
        assert params.boundary_type == ["periodic", "periodic", "periodic"]
        assert params.output_dir == "."
        assert params.with_mhd == False
        assert params.integrator_type == "euler"
        assert params.fluxer_type == "hll"

    def test_parameters_cell_sizes(self, sample_config):
        """Test cell size calculation"""
        params = Parameters(sample_config)

        expected_dx = 1.0 / 10  # lx / nx
        expected_dy = 0.8 / 8  # ly / ny
        expected_dz = 0.6 / 6  # lz / nz

        assert np.isclose(params.cell_sizes[0], expected_dx)
        assert np.isclose(params.cell_sizes[1], expected_dy)
        assert np.isclose(params.cell_sizes[2], expected_dz)

    def test_parameters_fixed_boundaries(self):
        """Test Parameters with fixed boundary conditions"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_fixed",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["fixed", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
        }

        params = Parameters(config)

        # Check that boundary values were set for fixed boundary
        assert len(params.boundary_value[0]) == 2  # left and right boundaries
        assert len(params.boundary_value[1]) == 0  # periodic, no boundary values
        assert len(params.boundary_value[2]) == 0  # periodic, no boundary values

    def test_parameters_with_source(self):
        """Test Parameters with source term"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "source": np.ones((5, 10, 8, 6)) * 0.1,
        }

        params = Parameters(config)

        assert params.source_data is not None
        assert params.source_data.shape == (5, 10, 8, 6)

    def test_parameters_with_gravity(self):
        """Test Parameters with gravity field"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_gravity",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "gravity": np.zeros((3, 10, 8, 6)),
        }

        params = Parameters(config)

        assert params.gravity_field is not None
        assert params.gravity_field.shape == (3, 10, 8, 6)

    def test_create_integrator_valid(self, sample_config):
        """Test creating valid integrator"""
        params = Parameters(sample_config)
        integrator = params.create_integrator()

        assert isinstance(integrator, Integrator)

    def test_create_integrator_invalid(self, sample_config):
        """Test creating integrator with invalid type"""
        sample_config["integrator"] = "invalid_integrator"
        params = Parameters(sample_config)

        with pytest.raises(KeyError):
            params.create_integrator()

    def test_create_fluxer_base(self, sample_config):
        """Test creating base flux calculator"""
        sample_config["fluxer"] = "base"
        params = Parameters(sample_config)
        fluxer = params.create_fluxer()

        assert isinstance(fluxer, FluxCalculator)

    def test_create_fluxer_hll(self, sample_config):
        """Test creating HLL fluxer"""
        sample_config["fluxer"] = "hll"
        params = Parameters(sample_config)
        fluxer = params.create_fluxer()

        assert isinstance(fluxer, HLLFluxer)

    def test_create_fluxer_invalid(self, sample_config):
        """Test creating fluxer with invalid type"""
        sample_config["fluxer"] = "invalid_fluxer"
        params = Parameters(sample_config)

        with pytest.raises(KeyError):
            params.create_fluxer()

    def test_create_source_none(self, sample_config):
        """Test creating source when none specified"""
        params = Parameters(sample_config)
        source = params.create_source()

        assert source is None

    def test_create_source_valid(self):
        """Test creating source with valid data"""
        source_data = np.ones((5, 10, 8, 6)) * 0.1
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "source": source_data,
        }

        params = Parameters(config)
        source = params.create_source()

        assert source is not None
        assert np.array_equal(
            source, source_data
        )  # Use the original source_data variable

    def test_create_source_invalid_shape(self):
        """Test creating source with invalid shape"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "source": np.ones((3, 5, 5, 5)),  # Wrong shape
        }

        params = Parameters(config)

        with pytest.raises(TypeError):
            params.create_source()

    def test_create_gravity_none(self, sample_config):
        """Test creating gravity when none specified"""
        params = Parameters(sample_config)
        gravity = params.create_gravity()

        assert gravity is None

    def test_create_gravity_valid(self):
        """Test creating gravity with valid field"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_gravity",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "gravity": np.zeros((3, 10, 8, 6)),
        }

        params = Parameters(config)
        gravity = params.create_gravity()

        assert gravity is not None
        from gawain.numerics import GravitySource

        assert isinstance(gravity, GravitySource)

    def test_create_gravity_invalid_shape(self):
        """Test creating gravity with invalid shape"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_gravity",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": ".",
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "gravity": np.zeros((2, 5, 5, 5)),  # Wrong shape
        }

        params = Parameters(config)

        with pytest.raises(TypeError):
            params.create_gravity()


class TestOutput:
    def test_output_initialization(
        self, temp_dir, sample_config, sample_solution_vector
    ):
        """Test Output initialization"""
        # Set output directory to temp directory
        sample_config["output_dir"] = temp_dir
        params = Parameters(sample_config)

        # The Output class creates an HDF5 file
        output = Output(params, sample_solution_vector)

        # Check that HDF5 file was created
        hdf5_file = os.path.join(temp_dir, "test_run.h5")
        assert os.path.exists(hdf5_file)

        # Check that initial dump was made (dump_no starts at 0, but increments)
        assert output.dump_no == 0  # Initial state

        # Check that the HDF5 file has expected structure
        import h5py

        with h5py.File(hdf5_file, "r") as f:
            assert "config" in f
            assert "solutions" in f
            assert "timestamps" in f
            assert "X" in f
            assert "Y" in f
            assert "Z" in f

        # Clean up
        output.close()

    def test_output_dump(
        self,
        temp_dir,
        sample_config,
        sample_solution_vector,
    ):
        """Test Output dump method"""
        sample_config["output_dir"] = temp_dir
        params = Parameters(sample_config)

        output = Output(params, sample_solution_vector)
        initial_dump_no = output.dump_no

        # Call dump method
        output.dump(sample_solution_vector, time=0.1)

        # Check that dump number was incremented
        assert output.dump_no == initial_dump_no + 1

        # Check that data was written to HDF5 file
        hdf5_file = os.path.join(temp_dir, "test_run.h5")
        assert os.path.exists(hdf5_file)

        # Check that the file contains the correct data
        import h5py

        with h5py.File(hdf5_file, "r") as f:
            solutions = f["solutions"][:]
            timestamps = f["timestamps"][:]

            # Should have 2 entries now (initial + 1 dump)
            assert solutions.shape[0] == 2
            assert timestamps.shape[0] == 2

            # Check the dumped data
            assert np.array_equal(solutions[1], sample_solution_vector.data)
            assert timestamps[1] == 0.1

        # Clean up
        output.close()

    def test_output_with_source(self, temp_dir, sample_solution_vector):
        """Test Output with source data"""
        # Create mesh grids
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 0.8, 8)
        z = np.linspace(0, 0.6, 6)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
            "mesh_grid": (X, Y, Z),
            "t_max": 1.0,
            "n_dumps": 5,
            "adi_idx": 1.4,
            "initial_condition": np.ones((5, 10, 8, 6)),
            "boundary_type": ["periodic", "periodic", "periodic"],
            "output_dir": temp_dir,
            "with_mhd": False,
            "integrator": "euler",
            "fluxer": "hll",
            "source": np.ones((5, 10, 8, 6)) * 0.1,
        }

        params = Parameters(config)
        output = Output(params, sample_solution_vector)

        # Check that source function field was saved in HDF5 file
        hdf5_file = os.path.join(temp_dir, "test_source.h5")
        assert os.path.exists(hdf5_file)

        # Verify the source data is correct
        import h5py

        with h5py.File(hdf5_file, "r") as f:
            assert "source_function_field" in f
            saved_source = f["source_function_field"][:]
            expected_source = np.ones((5, 10, 8, 6)) * 0.1
            assert np.allclose(saved_source, expected_source)

        # Clean up
        output.close()


class TestReader:
    def test_reader_initialization_hydro(self, temp_dir):
        """Test Reader initialization for hydro simulation"""
        # Create a mock HDF5 file for testing
        import h5py

        hdf5_file = os.path.join(temp_dir, "test_simulation.h5")

        with h5py.File(hdf5_file, "w") as f:
            # Create mock configuration
            config_group = f.create_group("config")
            config_group.attrs["with_mhd"] = False
            config_group.attrs["mesh_shape"] = [10, 8, 6]
            config_group.attrs["variables"] = [
                "density",
                "xmomentum",
                "ymomentum",
                "zmomentum",
                "energy",
            ]

            # Create mock grid data
            x = np.linspace(0, 1, 10)
            y = np.linspace(0, 1, 8)
            z = np.linspace(0, 1, 6)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            f.create_dataset("X", data=X)
            f.create_dataset("Y", data=Y)
            f.create_dataset("Z", data=Z)

            # Create mock timestamps
            f.create_dataset("timestamps", data=np.array([0.0, 0.1, 0.2]))

            # Create mock solution data
            solutions = np.random.rand(3, 5, 10, 8, 6)  # 3 timesteps, 5 variables
            f.create_dataset("solutions", data=solutions)

        reader = Reader(hdf5_file)

        assert reader.run_config["with_mhd"] == False
        assert reader.data_dim == 0  # No dimension with size 1 (3D simulation)

        # Check that data was loaded
        assert len(reader.data) == 5

    @pytest.mark.skip(
        reason="Reader tests require complex HDF5 setup, focus on core functionality"
    )
    def test_reader_features(self):
        """Placeholder for Reader functionality tests"""
        # Reader tests would require extensive HDF5 file setup that mirrors
        # the actual output format. For now, we focus on Parameters and Output testing.
        pass


class TestIOIntegration:
    def test_parameters_output_integration(
        self, temp_dir, sample_config, sample_solution_vector
    ):
        """Test integration between Parameters and Output"""
        sample_config["output_dir"] = temp_dir
        params = Parameters(sample_config)

        output = Output(params, sample_solution_vector)

        # Check that HDF5 file was created and contains proper config
        hdf5_file = os.path.join(temp_dir, "test_run.h5")
        assert os.path.exists(hdf5_file)

        # Load and verify the saved config from HDF5 attributes
        import h5py

        with h5py.File(hdf5_file, "r") as f:
            config_attrs = dict(f["config"].attrs)

        # Check that config was properly processed and stored
        assert config_attrs["run_name"] == "test_run"
        assert config_attrs["cfl"] == 0.5
        assert config_attrs["with_mhd"] == False
        assert list(config_attrs["variables"]) == [
            "density",
            "xmomentum",
            "ymomentum",
            "zmomentum",
            "energy",
        ]

        # Clean up
        output.close()
