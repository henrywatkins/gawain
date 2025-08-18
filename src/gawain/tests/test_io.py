import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from gawain.fluxes import (FluxCalculator, HLLFluxer, LaxFriedrichsFluxer,
                           LaxWendroffFluxer)
from gawain.integrators import Integrator
from gawain.io import Output, Parameters, Reader
from gawain.numerics import MHDSolutionVector, SolutionVector


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing"""
    config = {
        "run_name": "test_run",
        "cfl": 0.5,
        "mesh_shape": (10, 8, 6),
        "mesh_size": (1.0, 0.8, 0.6),
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
    config = {
        "run_name": "test_mhd_run",
        "cfl": 0.5,
        "mesh_shape": (10, 8, 6),
        "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_fixed",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_gravity",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_gravity",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
        config = {
            "run_name": "test_gravity",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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
    @patch("os.mkdir")
    @patch("os.path.exists")
    def test_output_initialization(
        self, mock_exists, mock_mkdir, temp_dir, sample_config, sample_solution_vector
    ):
        """Test Output initialization"""
        mock_exists.return_value = False

        # Set output directory to temp directory
        sample_config["output_dir"] = temp_dir
        params = Parameters(sample_config)

        with (
            patch("numpy.save") as mock_save,
            patch("builtins.open", create=True) as mock_open,
            patch("json.dump") as mock_json_dump,
        ):

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            output = Output(params, sample_solution_vector)

            # Check that directory was created
            expected_dir = os.path.join(temp_dir, "test_run")
            mock_mkdir.assert_called_once_with(expected_dir)

            # Check that initial dump was made
            assert output.dump_no == 1  # Should be incremented after initial dump

            # Check that config was saved
            mock_json_dump.assert_called_once()

    @patch("os.mkdir")
    @patch("os.path.exists")
    @patch("numpy.save")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_output_dump(
        self,
        mock_json,
        mock_open,
        mock_save,
        mock_exists,
        mock_mkdir,
        temp_dir,
        sample_config,
        sample_solution_vector,
    ):
        """Test Output dump method"""
        mock_exists.return_value = False
        sample_config["output_dir"] = temp_dir
        params = Parameters(sample_config)

        output = Output(params, sample_solution_vector)
        initial_dump_no = output.dump_no

        # Call dump method
        output.dump(sample_solution_vector)

        # Check that dump number was incremented
        assert output.dump_no == initial_dump_no + 1

        # Check that numpy save was called with correct filename
        expected_filename = os.path.join(
            temp_dir, "test_run", f"gawain_output_{initial_dump_no}.npy"
        )
        mock_save.assert_called_with(expected_filename, sample_solution_vector.data)

    @patch("os.mkdir")
    @patch("os.path.exists")
    def test_output_with_source(
        self, mock_exists, mock_mkdir, temp_dir, sample_solution_vector
    ):
        """Test Output with source data"""
        mock_exists.return_value = False

        config = {
            "run_name": "test_source",
            "cfl": 0.5,
            "mesh_shape": (10, 8, 6),
            "mesh_size": (1.0, 0.8, 0.6),
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

        with (
            patch("numpy.save") as mock_save,
            patch("builtins.open", create=True) as mock_open,
            patch("json.dump") as mock_json_dump,
        ):

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            output = Output(params, sample_solution_vector)

            # Check that source function field was saved
            source_call_found = False
            for call in mock_save.call_args_list:
                if "source_function_field.npy" in call[0][0]:
                    source_call_found = True
                    break
            assert source_call_found


class TestReader:
    def test_reader_initialization_hydro(self, temp_dir):
        """Test Reader initialization for hydro simulation"""
        run_dir = os.path.join(temp_dir, "test_run")
        os.makedirs(run_dir)

        # Create mock config file
        config = {
            "with_mhd": False,
            "mesh_shape": [10, 8, 6],  # No dimension with size 1, so data_dim = 0
            "n_dumps": 3,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files
        for i in range(3):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.random.rand(5, 10, 8, 6)  # 5 variables for hydro
            np.save(filename, data)

        reader = Reader(run_dir)

        assert reader.run_config["with_mhd"] == False
        assert len(reader.variables) == 5
        assert reader.variables == [
            "density",
            "xmomentum",
            "ymomentum",
            "zmomentum",
            "energy",
        ]
        assert reader.data_dim == 0  # No dimension with size 1 (3D simulation)

        # Check that data was loaded
        for var in reader.variables:
            assert var in reader.data
            assert reader.data[var].shape == (3, 10, 8, 6)  # 3 timesteps

    def test_reader_initialization_mhd(self, temp_dir):
        """Test Reader initialization for MHD simulation"""
        run_dir = os.path.join(temp_dir, "test_run_mhd")
        os.makedirs(run_dir)

        # Create mock config file
        config = {"with_mhd": True, "mesh_shape": [10, 8, 6], "n_dumps": 2}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files
        for i in range(2):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.random.rand(8, 10, 8, 6)  # 8 variables for MHD
            np.save(filename, data)

        reader = Reader(run_dir)

        assert reader.run_config["with_mhd"] == True
        assert len(reader.variables) == 8
        expected_vars = [
            "density",
            "xmomentum",
            "ymomentum",
            "zmomentum",
            "energy",
            "xmag",
            "ymag",
            "zmag",
        ]
        assert reader.variables == expected_vars

        # Check that data was loaded
        for var in reader.variables:
            assert var in reader.data
            assert reader.data[var].shape == (2, 10, 8, 6)  # 2 timesteps

    def test_reader_1d_data_dimension(self, temp_dir):
        """Test Reader with 1D data"""
        run_dir = os.path.join(temp_dir, "test_1d")
        os.makedirs(run_dir)

        # Create mock config file for 1D simulation
        config = {
            "with_mhd": False,
            "mesh_shape": [100, 1, 1],  # 1D simulation
            "n_dumps": 2,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files
        for i in range(2):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.random.rand(5, 100, 1, 1)
            np.save(filename, data)

        reader = Reader(run_dir)

        assert reader.data_dim == 2  # Two dimensions with size 1

    def test_reader_2d_data_dimension(self, temp_dir):
        """Test Reader with 2D data"""
        run_dir = os.path.join(temp_dir, "test_2d")
        os.makedirs(run_dir)

        # Create mock config file for 2D simulation
        config = {
            "with_mhd": False,
            "mesh_shape": [50, 50, 1],  # 2D simulation
            "n_dumps": 2,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files
        for i in range(2):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.random.rand(5, 50, 50, 1)
            np.save(filename, data)

        reader = Reader(run_dir)

        assert reader.data_dim == 1  # One dimension with size 1

    def test_reader_get_data(self, temp_dir):
        """Test Reader get_data method"""
        run_dir = os.path.join(temp_dir, "test_get_data")
        os.makedirs(run_dir)

        config = {"with_mhd": False, "mesh_shape": [10, 8, 6], "n_dumps": 2}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files with known data
        test_data = []
        for i in range(2):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.ones((5, 10, 8, 6)) * (
                i + 1
            )  # Different values for each timestep
            test_data.append(data)
            np.save(filename, data)

        reader = Reader(run_dir)

        density_data = reader.get_data("density")

        assert density_data.shape == (2, 10, 8, 6)
        assert np.allclose(density_data[0], 1.0)  # First timestep
        assert np.allclose(density_data[1], 2.0)  # Second timestep

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_reader_plot_1d(self, mock_savefig, mock_show, temp_dir):
        """Test Reader plot method for 1D data"""
        run_dir = os.path.join(temp_dir, "test_plot_1d")
        os.makedirs(run_dir)

        config = {"with_mhd": False, "mesh_shape": [20, 1, 1], "n_dumps": 2}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files
        for i in range(2):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.random.rand(5, 20, 1, 1)
            np.save(filename, data)

        reader = Reader(run_dir)

        # Test 1D plot
        reader.plot("density", timesteps=[0, 1])

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_reader_plot_2d(self, mock_savefig, mock_show, temp_dir):
        """Test Reader plot method for 2D data"""
        run_dir = os.path.join(temp_dir, "test_plot_2d")
        os.makedirs(run_dir)

        config = {"with_mhd": False, "mesh_shape": [10, 10, 1], "n_dumps": 2}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output files
        for i in range(2):
            filename = os.path.join(run_dir, f"gawain_output_{i}.npy")
            data = np.random.rand(5, 10, 10, 1)
            np.save(filename, data)

        reader = Reader(run_dir)

        # Test 2D plot
        reader.plot("density", timesteps=[0, 1])

        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_reader_plot_3d_error(self, mock_print, mock_show, temp_dir):
        """Test Reader plot method for 3D data (should print error)"""
        run_dir = os.path.join(temp_dir, "test_plot_3d")
        os.makedirs(run_dir)

        config = {"with_mhd": False, "mesh_shape": [5, 5, 5], "n_dumps": 1}  # 3D data
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create mock output file
        filename = os.path.join(run_dir, "gawain_output_0.npy")
        data = np.random.rand(5, 5, 5, 5)
        np.save(filename, data)

        reader = Reader(run_dir)

        # Test 3D plot (should print error message)
        reader.plot("density")

        mock_print.assert_called_with(
            "plot() only supports visualisation of 1D and 2D data"
        )
        mock_show.assert_not_called()


class TestIOIntegration:
    def test_parameters_output_integration(
        self, temp_dir, sample_config, sample_solution_vector
    ):
        """Test integration between Parameters and Output"""
        sample_config["output_dir"] = temp_dir
        params = Parameters(sample_config)

        with patch("os.mkdir"), patch("os.path.exists", return_value=False):
            with (
                patch("numpy.save") as mock_save,
                patch("builtins.open", create=True) as mock_open,
                patch("json.dump") as mock_json_dump,
            ):

                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                output = Output(params, sample_solution_vector)

                # Check that config was properly processed for JSON serialization
                # (initial_condition, source, gravity removed from config)
                mock_json_dump.assert_called_once()
                saved_config = mock_json_dump.call_args[0][0]
                assert "initial_condition" not in saved_config
                assert "source" not in saved_config
                assert "gravity" not in saved_config
