from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from gawain.fluxes import FluxCalculator
from gawain.integrators import Integrator
from gawain.numerics import GravitySource, MHDSolutionVector, SolutionVector


@pytest.fixture
def mock_parameters():
    """Mock Parameters object for testing integrator"""
    params = Mock()
    params.with_mhd = False

    # Mock fluxer creation
    fluxer = Mock(spec=FluxCalculator)
    fluxer.set_flux_function = Mock()
    fluxer.calculate_flux_divergence = Mock()
    params.create_fluxer = Mock(return_value=fluxer)

    # Mock source creation
    params.create_source = Mock(return_value=None)

    # Mock gravity creation
    params.create_gravity = Mock(return_value=None)

    return params


@pytest.fixture
def mock_parameters_with_source():
    """Mock Parameters object with source term"""
    params = Mock()
    params.with_mhd = False

    # Mock fluxer creation
    fluxer = Mock(spec=FluxCalculator)
    fluxer.set_flux_function = Mock()
    fluxer.calculate_flux_divergence = Mock()
    params.create_fluxer = Mock(return_value=fluxer)

    # Mock source creation - return simple source array
    source_data = np.ones((5, 10, 8, 6))
    params.create_source = Mock(return_value=source_data)

    # Mock gravity creation
    params.create_gravity = Mock(return_value=None)

    return params


@pytest.fixture
def mock_parameters_with_gravity():
    """Mock Parameters object with gravity source"""
    params = Mock()
    params.with_mhd = False

    # Mock fluxer creation
    fluxer = Mock(spec=FluxCalculator)
    fluxer.set_flux_function = Mock()
    fluxer.calculate_flux_divergence = Mock()
    params.create_fluxer = Mock(return_value=fluxer)

    # Mock source creation
    params.create_source = Mock(return_value=None)

    # Mock gravity creation
    gravity_field = np.zeros((3, 10, 8, 6))
    gravity_field[2] = -9.8  # gravity in z direction
    gravity = GravitySource(gravity_field)
    params.create_gravity = Mock(return_value=gravity)

    return params


@pytest.fixture
def sample_solution_vector():
    """Create a sample solution vector for testing"""
    sv = SolutionVector()
    nx, ny, nz = 10, 8, 6
    data = np.zeros((5, nx, ny, nz))

    # Set realistic hydrodynamic values
    data[0] = 1.0  # density
    data[1] = 0.1  # x-momentum
    data[2] = 0.0  # y-momentum
    data[3] = 0.0  # z-momentum
    data[4] = 2.5  # energy

    sv.data = data
    sv.adi_idx = 1.4
    sv.dx, sv.dy, sv.dz = 0.1, 0.1, 0.1
    sv.timestep = 0.01

    return sv


@pytest.fixture
def sample_mhd_solution_vector():
    """Create a sample MHD solution vector for testing"""
    mhd_sv = MHDSolutionVector()
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

    mhd_sv.data = data
    mhd_sv.adi_idx = 1.4
    mhd_sv.dx, mhd_sv.dy, mhd_sv.dz = 0.1, 0.1, 0.1
    mhd_sv.timestep = 0.01

    return mhd_sv


class TestIntegrator:
    def test_integrator_initialization(self, mock_parameters):
        """Test Integrator initialization"""
        integrator = Integrator(mock_parameters)

        # Check that components were created
        mock_parameters.create_fluxer.assert_called_once()
        mock_parameters.create_source.assert_called_once()
        mock_parameters.create_gravity.assert_called_once()

        # Check that fluxer was configured
        integrator.fluxer.set_flux_function.assert_called_once_with(
            mock_parameters.with_mhd
        )

        # Check attributes
        assert integrator.fluxer is not None
        assert integrator.source is None  # No source in basic mock
        assert integrator.gravity is None  # No gravity in basic mock

    def test_integrator_initialization_with_mhd(self):
        """Test Integrator initialization with MHD"""
        params = Mock()
        params.with_mhd = True

        fluxer = Mock(spec=FluxCalculator)
        fluxer.set_flux_function = Mock()
        params.create_fluxer = Mock(return_value=fluxer)
        params.create_source = Mock(return_value=None)
        params.create_gravity = Mock(return_value=None)

        integrator = Integrator(params)

        # Check that fluxer was configured for MHD
        integrator.fluxer.set_flux_function.assert_called_once_with(True)

    def test_integrate_basic(self, mock_parameters, sample_solution_vector):
        """Test basic integration without sources"""
        integrator = Integrator(mock_parameters)

        # Set up flux divergence return value
        flux_div = np.ones_like(sample_solution_vector.data) * 0.1
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        # Store original data
        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Check that flux divergence was calculated
        integrator.fluxer.calculate_flux_divergence.assert_called_once_with(
            sample_solution_vector
        )

        # Check that solution was updated
        expected_data = original_data + sample_solution_vector.timestep * flux_div
        assert np.allclose(result.data, expected_data)

        # Check that the same object is returned
        assert result is sample_solution_vector

    def test_integrate_with_source(
        self, mock_parameters_with_source, sample_solution_vector
    ):
        """Test integration with source term"""
        integrator = Integrator(mock_parameters_with_source)

        # Set up flux divergence return value
        flux_div = np.ones_like(sample_solution_vector.data) * 0.1
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        # Store original data
        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Verify that integrator has source
        assert integrator.source is not None

        # Verify that integration completed and modified the data
        assert not np.array_equal(result.data, original_data)

    def test_integrate_with_gravity(
        self, mock_parameters_with_gravity, sample_solution_vector
    ):
        """Test integration with gravity source"""
        integrator = Integrator(mock_parameters_with_gravity)

        # Set up flux divergence return value
        flux_div = np.ones_like(sample_solution_vector.data) * 0.1
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        # Store original data
        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Check that flux divergence was calculated
        integrator.fluxer.calculate_flux_divergence.assert_called_once_with(
            sample_solution_vector
        )

        # Verify that integrator has gravity
        assert integrator.gravity is not None
        from gawain.numerics import GravitySource

        assert isinstance(integrator.gravity, GravitySource)

        # Verify that integration completed and modified the data
        assert not np.array_equal(result.data, original_data)

    def test_integrate_with_all_sources(self):
        """Test integration with both regular source and gravity"""
        params = Mock()
        params.with_mhd = False

        # Mock fluxer
        fluxer = Mock(spec=FluxCalculator)
        fluxer.set_flux_function = Mock()
        flux_div = np.ones((5, 10, 8, 6)) * 0.1
        fluxer.calculate_flux_divergence = Mock(return_value=flux_div)
        params.create_fluxer = Mock(return_value=fluxer)

        # Mock regular source
        source_data = np.ones((5, 10, 8, 6)) * 0.05
        params.create_source = Mock(return_value=source_data)

        # Mock gravity
        gravity_field = np.zeros((3, 10, 8, 6))
        gravity_field[2] = -9.8
        gravity = GravitySource(gravity_field)
        params.create_gravity = Mock(return_value=gravity)

        integrator = Integrator(params)

        # Create solution vector
        sv = SolutionVector()
        sv.data = np.ones((5, 10, 8, 6))
        sv.data[0] = 1.0  # density
        sv.timestep = 0.01

        original_data = sv.data.copy()

        # Perform integration
        result = integrator.integrate(sv)

        # Check that all terms were included
        assert not np.allclose(result.data, original_data)

        # The result should include flux divergence, source, and gravity
        # This is a qualitative check - exact values depend on gravity calculation

    def test_integrate_conservation_properties(
        self, mock_parameters, sample_solution_vector
    ):
        """Test that integration preserves basic physical properties"""
        integrator = Integrator(mock_parameters)

        # Set up a flux divergence that should conserve mass in uniform case
        flux_div = np.zeros_like(sample_solution_vector.data)
        # Add some non-zero momentum flux divergence
        flux_div[1:4] = 0.1
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Mass should be conserved (no mass flux divergence)
        assert np.allclose(result.data[0], original_data[0])

        # Momentum should change
        assert not np.allclose(result.data[1:4], original_data[1:4])

    def test_integrate_mhd_solution(self, sample_mhd_solution_vector):
        """Test integration with MHD solution vector"""
        params = Mock()
        params.with_mhd = True

        # Mock fluxer for MHD
        fluxer = Mock(spec=FluxCalculator)
        fluxer.set_flux_function = Mock()
        flux_div = np.ones((8, 10, 8, 6)) * 0.1  # 8 variables for MHD
        fluxer.calculate_flux_divergence = Mock(return_value=flux_div)
        params.create_fluxer = Mock(return_value=fluxer)
        params.create_source = Mock(return_value=None)
        params.create_gravity = Mock(return_value=None)

        integrator = Integrator(params)

        original_data = sample_mhd_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_mhd_solution_vector)

        # Check that MHD flux function was set
        integrator.fluxer.set_flux_function.assert_called_once_with(True)

        # Check that all 8 MHD variables were updated
        assert result.data.shape[0] == 8
        expected_data = original_data + sample_mhd_solution_vector.timestep * flux_div
        assert np.allclose(result.data, expected_data)

    def test_integrate_timestep_scaling(self, mock_parameters, sample_solution_vector):
        """Test that integration properly scales with timestep"""
        integrator = Integrator(mock_parameters)

        flux_div = np.ones_like(sample_solution_vector.data)
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        # Test with different timesteps
        timesteps = [0.001, 0.01, 0.1]

        for dt in timesteps:
            sv = SolutionVector()
            sv.data = np.ones((5, 10, 8, 6))
            sv.timestep = dt

            original_data = sv.data.copy()

            result = integrator.integrate(sv)

            change = result.data - original_data
            expected_change = dt * flux_div

            assert np.allclose(change, expected_change)

    def test_integrate_zero_timestep(self, mock_parameters, sample_solution_vector):
        """Test integration with zero timestep"""
        integrator = Integrator(mock_parameters)

        flux_div = np.ones_like(sample_solution_vector.data)
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        sample_solution_vector.timestep = 0.0
        original_data = sample_solution_vector.data.copy()

        result = integrator.integrate(sample_solution_vector)

        # With zero timestep, data should not change
        assert np.allclose(result.data, original_data)

    def test_integrate_error_handling(self, mock_parameters, sample_solution_vector):
        """Test integration handles potential errors gracefully"""
        integrator = Integrator(mock_parameters)

        # Test with flux divergence containing NaN
        flux_div = np.ones_like(sample_solution_vector.data)
        flux_div[0, 0, 0, 0] = np.nan
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        result = integrator.integrate(sample_solution_vector)

        # Check that NaN propagated as expected
        assert np.isnan(result.data[0, 0, 0, 0])

    def test_integrator_components_interaction(
        self, mock_parameters_with_source, sample_solution_vector
    ):
        """Test interaction between different integrator components"""
        integrator = Integrator(mock_parameters_with_source)

        # Verify all components are set up correctly
        assert integrator.fluxer is not None
        assert integrator.source is not None
        assert integrator.gravity is None

        # Test that fluxer and source interact correctly
        flux_div = np.ones_like(sample_solution_vector.data) * 0.1
        integrator.fluxer.calculate_flux_divergence.return_value = flux_div

        original_data = sample_solution_vector.data.copy()
        result = integrator.integrate(sample_solution_vector)

        # Verify that integration ran and modified the solution
        assert not np.array_equal(result.data, original_data)

        # Verify the fluxer was called
        integrator.fluxer.calculate_flux_divergence.assert_called_once_with(
            sample_solution_vector
        )
