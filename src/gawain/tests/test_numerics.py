from unittest.mock import Mock

import numpy as np
import pytest

from gawain.numerics import (BoundarySetter, Clock, GravitySource,
                             MHDSolutionVector, SolutionVector)


@pytest.fixture
def mock_parameters():
    """Mock Parameters object for testing"""
    params = Mock()
    params.t_max = 1.0
    params.n_outputs = 10
    params.boundary_type = ["periodic", "periodic", "periodic"]
    params.boundary_value = [[], [], []]
    params.cell_sizes = (0.1, 0.1, 0.1)
    params.adi_idx = 1.4
    params.cfl = 0.5
    params.initial_condition = np.ones((5, 10, 10, 10))
    return params


@pytest.fixture
def sample_solution_data():
    """Sample solution vector data"""
    # Create a simple 3D mesh with 5 variables
    nx, ny, nz = 10, 8, 6
    data = np.zeros((5, nx, ny, nz))

    # Set realistic hydrodynamic values
    data[0] = 1.0  # density
    data[1] = 0.1  # x-momentum
    data[2] = 0.0  # y-momentum
    data[3] = 0.0  # z-momentum
    data[4] = 2.5  # energy

    return data


@pytest.fixture
def sample_mhd_data():
    """Sample MHD solution vector data"""
    # Create a simple 3D mesh with 8 variables
    nx, ny, nz = 10, 8, 6
    data = np.zeros((8, nx, ny, nz))

    # Set realistic MHD values
    data[0] = 1.0  # density
    data[1] = 0.1  # x-momentum
    data[2] = 0.0  # y-momentum
    data[3] = 0.0  # z-momentum
    data[4] = 3.0  # energy
    data[5] = 0.1  # x-magnetic field
    data[6] = 0.0  # y-magnetic field
    data[7] = 0.0  # z-magnetic field

    return data


class TestClock:
    def test_clock_initialization(self, mock_parameters):
        """Test Clock initialization"""
        clock = Clock(mock_parameters)

        assert clock.current_time == 0.0
        assert clock.end_time == mock_parameters.t_max
        assert clock.next_output_time == 0.0
        assert clock.output_spacing == mock_parameters.t_max / mock_parameters.n_outputs

    def test_clock_is_end_false(self, mock_parameters):
        """Test is_end returns False when simulation not finished"""
        clock = Clock(mock_parameters)
        assert not clock.is_end()

    def test_clock_is_end_true(self, mock_parameters):
        """Test is_end returns True when simulation finished"""
        clock = Clock(mock_parameters)
        clock.current_time = mock_parameters.t_max + 0.1
        assert clock.is_end()

    def test_clock_tick(self, mock_parameters):
        """Test clock tick updates current time"""
        clock = Clock(mock_parameters)
        dt = 0.01
        initial_time = clock.current_time
        clock.tick(dt)
        assert clock.current_time == initial_time + dt

    def test_clock_is_output_true(self, mock_parameters):
        """Test is_output returns True when output time reached"""
        clock = Clock(mock_parameters)
        clock.current_time = clock.output_spacing
        assert clock.is_output()

    def test_clock_is_output_false(self, mock_parameters):
        """Test is_output returns False when output time not reached"""
        clock = Clock(mock_parameters)
        clock.current_time = clock.output_spacing / 2

        # Reset next_output_time to ensure we're testing the right condition
        clock.next_output_time = clock.output_spacing
        assert not clock.is_output()

    def test_clock_duration(self, mock_parameters):
        """Test duration calculation"""
        clock = Clock(mock_parameters)
        duration = clock.duration()
        assert isinstance(duration, float)
        assert duration >= 0


class TestSolutionVector:
    def test_solution_vector_initialization(self):
        """Test SolutionVector initialization"""
        sv = SolutionVector()

        assert sv.data is None
        assert sv.boundary_type is None
        assert sv.boundary_value is None
        assert sv.boundsetter is None
        assert sv.dx is None and sv.dy is None and sv.dz is None
        assert sv.adi_idx == 1.4
        assert sv.timestep == 0.0001
        assert sv.cfl == 0.1
        assert sv.variable_names == [
            "density",
            "xmomentum",
            "ymomentum",
            "zmomentum",
            "energy",
        ]

    def test_set_state(self, mock_parameters):
        """Test set_state method"""
        sv = SolutionVector()
        sv.set_state(mock_parameters)

        assert sv.boundary_type == mock_parameters.boundary_type
        assert sv.boundary_value == mock_parameters.boundary_value
        assert sv.dx == mock_parameters.cell_sizes[0]
        assert sv.dy == mock_parameters.cell_sizes[1]
        assert sv.dz == mock_parameters.cell_sizes[2]
        assert sv.adi_idx == mock_parameters.adi_idx
        assert sv.cfl == mock_parameters.cfl
        assert np.array_equal(sv.data, mock_parameters.initial_condition)
        assert sv.boundsetter is not None

    def test_copy(self, mock_parameters, sample_solution_data):
        """Test copy method"""
        sv = SolutionVector()
        sv.set_state(mock_parameters)
        sv.data = sample_solution_data

        copy_sv = sv.copy()

        assert np.array_equal(copy_sv.data, sv.data)
        assert copy_sv.boundary_type == sv.boundary_type
        assert copy_sv.dx == sv.dx
        assert copy_sv.adi_idx == sv.adi_idx

    def test_get_variable(self, sample_solution_data):
        """Test get_variable method"""
        sv = SolutionVector()
        sv.data = sample_solution_data

        density = sv.get_variable("density")
        assert np.array_equal(density, sample_solution_data[0])

        energy = sv.get_variable("energy")
        assert np.array_equal(energy, sample_solution_data[4])

    def test_field_accessors(self, sample_solution_data):
        """Test field accessor methods"""
        sv = SolutionVector()
        sv.data = sample_solution_data

        assert np.array_equal(sv.dens(), sample_solution_data[0])
        assert np.array_equal(sv.momX(), sample_solution_data[1])
        assert np.array_equal(sv.momY(), sample_solution_data[2])
        assert np.array_equal(sv.momZ(), sample_solution_data[3])
        assert np.array_equal(sv.energy(), sample_solution_data[4])

    def test_velocity_calculations(self, sample_solution_data):
        """Test velocity calculation methods"""
        sv = SolutionVector()
        sv.data = sample_solution_data

        expected_velX = sample_solution_data[1] / sample_solution_data[0]
        expected_velY = sample_solution_data[2] / sample_solution_data[0]
        expected_velZ = sample_solution_data[3] / sample_solution_data[0]

        assert np.allclose(sv.velX(), expected_velX)
        assert np.allclose(sv.velY(), expected_velY)
        assert np.allclose(sv.velZ(), expected_velZ)

    def test_pressure_calculation(self, sample_solution_data):
        """Test pressure calculation"""
        sv = SolutionVector()
        sv.data = sample_solution_data
        sv.adi_idx = 1.4

        pressure = sv.pressure()

        # Check that pressure is positive and finite
        assert np.all(pressure >= 0)
        assert np.all(np.isfinite(pressure))

    def test_sound_speed_calculation(self, sample_solution_data):
        """Test sound speed calculation"""
        sv = SolutionVector()
        sv.data = sample_solution_data
        sv.adi_idx = 1.4

        cs = sv.sound_speed()

        # Check that sound speed is positive and finite
        assert np.all(cs > 0)
        assert np.all(np.isfinite(cs))

    def test_wave_speeds_x(self, sample_solution_data):
        """Test wave speed calculations in x direction"""
        sv = SolutionVector()
        sv.data = sample_solution_data
        sv.adi_idx = 1.4

        lambda_min, lambda_max = sv.calculate_min_max_wave_speeds_X()

        # Min should be less than max
        assert np.all(lambda_min <= lambda_max)
        assert np.all(np.isfinite(lambda_min))
        assert np.all(np.isfinite(lambda_max))

    def test_wave_speeds_y(self, sample_solution_data):
        """Test wave speed calculations in y direction"""
        sv = SolutionVector()
        sv.data = sample_solution_data
        sv.adi_idx = 1.4

        lambda_min, lambda_max = sv.calculate_min_max_wave_speeds_Y()

        # Min should be less than max
        assert np.all(lambda_min <= lambda_max)
        assert np.all(np.isfinite(lambda_min))
        assert np.all(np.isfinite(lambda_max))

    def test_calculate_timestep(self, mock_parameters, sample_solution_data):
        """Test timestep calculation"""
        sv = SolutionVector()
        sv.set_state(mock_parameters)
        sv.data = sample_solution_data

        dt = sv.calculate_timestep()

        assert dt > 0
        assert np.isfinite(dt)
        assert sv.timestep == dt

    def test_update_method(self, sample_solution_data):
        """Test update method"""
        sv = SolutionVector()
        sv.data = sample_solution_data.copy()
        sv.timestep = 0.01

        update_array = np.ones_like(sample_solution_data)
        original_data = sv.data.copy()

        sv.update(update_array)

        expected = original_data + sv.timestep * update_array
        assert np.allclose(sv.data, expected)


class TestMHDSolutionVector:
    def test_mhd_initialization(self):
        """Test MHDSolutionVector initialization"""
        mhd_sv = MHDSolutionVector()

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
        assert mhd_sv.variable_names == expected_vars

    def test_mhd_field_accessors(self, sample_mhd_data):
        """Test MHD field accessor methods"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data

        assert np.array_equal(mhd_sv.magX(), sample_mhd_data[5])
        assert np.array_equal(mhd_sv.magY(), sample_mhd_data[6])
        assert np.array_equal(mhd_sv.magZ(), sample_mhd_data[7])

    def test_magnetic_pressure(self, sample_mhd_data):
        """Test magnetic pressure calculation"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data

        mag_pressure = mhd_sv.magnetic_pressure()
        expected = 0.5 * mhd_sv.magTotalSqr()

        assert np.allclose(mag_pressure, expected)
        assert np.all(mag_pressure >= 0)

    def test_total_pressure(self, sample_mhd_data):
        """Test total pressure calculation"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data
        mhd_sv.adi_idx = 1.4

        total_pressure = mhd_sv.total_pressure()
        thermal_pressure = mhd_sv.pressure()
        magnetic_pressure = mhd_sv.magnetic_pressure()

        expected = thermal_pressure + magnetic_pressure
        assert np.allclose(total_pressure, expected)

    def test_alfven_speed(self, sample_mhd_data):
        """Test Alfven speed calculation"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data

        va = mhd_sv.alfven_speed()

        assert np.all(va >= 0)
        assert np.all(np.isfinite(va))

    def test_fast_magnetosonic_speed_x(self, sample_mhd_data):
        """Test fast magnetosonic speed in x direction"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data
        mhd_sv.adi_idx = 1.4

        cf = mhd_sv.fast_magnetosonic_speed_X()

        assert np.all(cf >= 0)
        assert np.all(np.isfinite(cf))

    def test_fast_magnetosonic_speed_y(self, sample_mhd_data):
        """Test fast magnetosonic speed in y direction"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data
        mhd_sv.adi_idx = 1.4

        cf = mhd_sv.fast_magnetosonic_speed_Y()

        assert np.all(cf >= 0)
        assert np.all(np.isfinite(cf))

    def test_mhd_wave_speeds(self, sample_mhd_data):
        """Test MHD wave speed calculations"""
        mhd_sv = MHDSolutionVector()
        mhd_sv.data = sample_mhd_data
        mhd_sv.adi_idx = 1.4

        lambda_min_x, lambda_max_x = mhd_sv.calculate_min_max_wave_speeds_X()
        lambda_min_y, lambda_max_y = mhd_sv.calculate_min_max_wave_speeds_Y()

        assert np.all(lambda_min_x <= lambda_max_x)
        assert np.all(lambda_min_y <= lambda_max_y)
        assert np.all(np.isfinite(lambda_min_x))
        assert np.all(np.isfinite(lambda_max_x))
        assert np.all(np.isfinite(lambda_min_y))
        assert np.all(np.isfinite(lambda_max_y))


class TestBoundarySetter:
    def test_boundary_setter_initialization(self):
        """Test BoundarySetter initialization"""
        boundary_types = ["periodic", "outflow", "fixed"]
        initial_values = np.ones((3, 10, 10, 10))

        bs = BoundarySetter(boundary_types, initial_values)

        assert bs.boundary_types == boundary_types
        assert np.array_equal(bs.initial_values, initial_values)

    def test_periodic_boundary(self, sample_solution_data):
        """Test periodic boundary conditions"""
        boundary_types = ["periodic", "periodic", "periodic"]
        bs = BoundarySetter(boundary_types, sample_solution_data)

        result = bs.set_stencil(sample_solution_data, axis=0, direction=1)

        # For periodic boundaries, should just be a roll
        expected = np.roll(sample_solution_data, 1, axis=1)
        assert np.array_equal(result, expected)

    def test_get_boundary_indices(self, sample_solution_data):
        """Test boundary index calculation"""
        boundary_types = ["outflow", "outflow", "outflow"]
        bs = BoundarySetter(boundary_types, sample_solution_data)

        shape = sample_solution_data.shape
        indices = bs.get_boundary_indices(axis=0, direction=1, shape=shape)

        # Check that we get proper indexing arrays
        assert len(indices) == 4  # variables, x, y, z indices


class TestGravitySource:
    def test_gravity_source_initialization(self):
        """Test GravitySource initialization"""
        gravity_field = np.zeros((3, 10, 10, 10))
        gravity_field[2] = -9.8  # gravity in z direction

        gs = GravitySource(gravity_field)

        assert np.array_equal(gs.field, gravity_field)

    def test_gravity_source_calculation(self, sample_solution_data):
        """Test gravity source term calculation"""
        # Create a simple gravity field
        shape = sample_solution_data.shape[1:]  # Get spatial dimensions
        gravity_field = np.zeros((3,) + shape)
        gravity_field[2] = -9.8  # gravity in z direction

        gs = GravitySource(gravity_field)

        # Create a mock solution vector
        sv = SolutionVector()
        sv.data = sample_solution_data

        source = gs.calculate_gravity_source(sv)

        # Check shape
        assert source.shape == sample_solution_data.shape

        # Check that momentum gets gravity source
        expected_momentum_z = sv.dens() * gravity_field[2]
        assert np.allclose(source[3], expected_momentum_z)

        # Check that energy gets work done by gravity
        expected_energy = (
            sv.momX() * gravity_field[0]
            + sv.momY() * gravity_field[1]
            + sv.momZ() * gravity_field[2]
        )
        assert np.allclose(source[4], expected_energy)
