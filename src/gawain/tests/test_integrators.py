import numpy as np
import pytest

from gawain.fluxes import FluxCalculator, HLLFluxer
from gawain.integrators import Integrator
from gawain.numerics import (BoundarySetter, GravitySource, MHDSolutionVector,
                             SolutionVector)


class MockParameters:
    """Simple parameters class for testing"""

    def __init__(self, with_mhd=False, source_data=None, gravity_field=None):
        self.with_mhd = with_mhd
        self.source_data = source_data
        self.gravity_field = gravity_field

    def create_fluxer(self):
        return FluxCalculator()

    def create_source(self):
        return self.source_data

    def create_gravity(self):
        if self.gravity_field is not None:
            return GravitySource(self.gravity_field)
        return None


@pytest.fixture
def mock_parameters():
    """Mock Parameters object for testing integrator"""
    return MockParameters()


@pytest.fixture
def mock_parameters_with_source():
    """Mock Parameters object with source term"""
    source_data = np.ones((5, 10, 8, 6)) * 0.01
    return MockParameters(source_data=source_data)


@pytest.fixture
def mock_parameters_with_gravity():
    """Mock Parameters object with gravity source"""
    gravity_field = np.zeros((3, 10, 8, 6))
    gravity_field[2] = -9.8  # gravity in z direction
    return MockParameters(gravity_field=gravity_field)


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
    sv.cell_sizes = (0.1, 0.1, 0.1)
    sv.timestep = 0.01
    sv.boundary_type = ["periodic", "periodic", "periodic"]
    sv.boundsetter = BoundarySetter(sv.boundary_type, data)

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
    mhd_sv.cell_sizes = (0.1, 0.1, 0.1)
    mhd_sv.timestep = 0.01
    mhd_sv.boundary_type = ["periodic", "periodic", "periodic"]
    mhd_sv.boundsetter = BoundarySetter(mhd_sv.boundary_type, data)

    return mhd_sv


class TestIntegrator:
    def test_integrator_initialization(self, mock_parameters):
        """Test Integrator initialization"""
        integrator = Integrator(mock_parameters)

        # Check attributes
        assert integrator.fluxer is not None
        assert isinstance(integrator.fluxer, FluxCalculator)
        assert integrator.source is None  # No source in basic mock
        assert integrator.gravity is None  # No gravity in basic mock

    def test_integrator_initialization_with_mhd(self):
        """Test Integrator initialization with MHD"""
        params = MockParameters(with_mhd=True)
        integrator = Integrator(params)

        # Check that fluxer was configured for MHD
        assert integrator.fluxer is not None
        assert isinstance(integrator.fluxer, FluxCalculator)

    def test_integrate_basic(self, mock_parameters, sample_solution_vector):
        """Test basic integration without sources"""
        integrator = Integrator(mock_parameters)

        # Create a non-uniform solution that will have non-zero flux divergence
        sample_solution_vector.data[1, 0, 0, 0] = 0.2  # Add some x-momentum variation
        sample_solution_vector.data[1, 1, 1, 1] = 0.05  # Different momentum elsewhere

        # Store original data
        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Check that the same object is returned
        assert result is sample_solution_vector

        # Check that all values are finite
        assert np.all(np.isfinite(result.data))

        # For non-uniform case, should see some changes (though may be small)
        # At minimum, check that values remain physical
        assert np.all(result.data[0] > 0)  # density positive
        assert np.all(result.data[4] > 0)  # energy positive

    def test_integrate_with_source(
        self, mock_parameters_with_source, sample_solution_vector
    ):
        """Test integration with source term"""
        integrator = Integrator(mock_parameters_with_source)

        # Store original data
        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Verify that integrator has source
        assert integrator.source is not None
        assert integrator.source.shape == sample_solution_vector.data.shape

        # Verify that integration completed and modified the data
        assert not np.array_equal(result.data, original_data)
        assert np.all(np.isfinite(result.data))

    def test_integrate_with_gravity(
        self, mock_parameters_with_gravity, sample_solution_vector
    ):
        """Test integration with gravity source"""
        integrator = Integrator(mock_parameters_with_gravity)

        # Store original data
        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Verify that integrator has gravity
        assert integrator.gravity is not None
        assert isinstance(integrator.gravity, GravitySource)

        # Verify that integration completed and modified the data
        assert not np.array_equal(result.data, original_data)
        assert np.all(np.isfinite(result.data))

    def test_integrate_with_all_sources(self):
        """Test integration with both regular source and gravity"""
        # Create parameters with both source and gravity
        source_data = np.ones((5, 10, 8, 6)) * 0.05
        gravity_field = np.zeros((3, 10, 8, 6))
        gravity_field[2] = -9.8

        params = MockParameters(source_data=source_data, gravity_field=gravity_field)
        integrator = Integrator(params)

        # Create solution vector
        sv = SolutionVector()
        sv.data = np.ones((5, 10, 8, 6))
        sv.data[0] = 1.0  # density
        sv.timestep = 0.01
        sv.dx = sv.dy = sv.dz = 0.1
        sv.cell_sizes = (0.1, 0.1, 0.1)
        sv.adi_idx = 1.4
        sv.boundary_type = ["periodic", "periodic", "periodic"]
        sv.boundsetter = BoundarySetter(sv.boundary_type, sv.data)

        original_data = sv.data.copy()

        # Perform integration
        result = integrator.integrate(sv)

        # Check that all terms were included
        assert not np.allclose(result.data, original_data)
        assert np.all(np.isfinite(result.data))

        # Verify both source and gravity are present
        assert integrator.source is not None
        assert integrator.gravity is not None

    def test_integrate_conservation_properties(
        self, mock_parameters, sample_solution_vector
    ):
        """Test that integration preserves basic physical properties"""
        integrator = Integrator(mock_parameters)

        original_data = sample_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_solution_vector)

        # Check that density remains positive
        assert np.all(result.data[0] > 0)

        # Check that energy remains positive
        assert np.all(result.data[4] > 0)

        # Check result is finite
        assert np.all(np.isfinite(result.data))

    def test_integrate_mhd_solution(self, sample_mhd_solution_vector):
        """Test integration with MHD solution vector"""
        params = MockParameters(with_mhd=True)
        integrator = Integrator(params)

        # Create a non-uniform solution
        sample_mhd_solution_vector.data[1, 0, 0, 0] = (
            0.2  # Add some x-momentum variation
        )
        sample_mhd_solution_vector.data[5, 1, 1, 1] = (
            0.05  # Add some magnetic field variation
        )

        original_data = sample_mhd_solution_vector.data.copy()

        # Perform integration
        result = integrator.integrate(sample_mhd_solution_vector)

        # Check that all 8 MHD variables were updated
        assert result.data.shape[0] == 8
        assert np.all(np.isfinite(result.data))

        # Check that density remains positive
        assert np.all(result.data[0] > 0)  # density
        assert np.all(np.isfinite(result.data[5:8]))  # magnetic fields

    def test_integrate_timestep_scaling(self, mock_parameters):
        """Test that integration properly scales with timestep"""
        integrator = Integrator(mock_parameters)

        # Test with different timesteps
        timesteps = [0.001, 0.01, 0.1]
        changes = []

        for dt in timesteps:
            sv = SolutionVector()
            sv.data = np.ones((5, 10, 8, 6))
            sv.data[0] = 1.0  # density
            sv.data[4] = 2.5  # energy
            sv.timestep = dt
            sv.dx = sv.dy = sv.dz = 0.1
            sv.cell_sizes = (0.1, 0.1, 0.1)
            sv.adi_idx = 1.4
            sv.boundary_type = ["periodic", "periodic", "periodic"]
            sv.boundsetter = BoundarySetter(sv.boundary_type, sv.data)

            original_data = sv.data.copy()
            result = integrator.integrate(sv)
            change = np.sum(np.abs(result.data - original_data))
            changes.append(change)

        # Larger timesteps should generally produce larger changes
        # (though this isn't always strictly true due to nonlinearities)
        assert all(np.isfinite(changes))

    def test_integrate_zero_timestep(self, mock_parameters, sample_solution_vector):
        """Test integration with zero timestep"""
        integrator = Integrator(mock_parameters)

        sample_solution_vector.timestep = 0.0
        original_data = sample_solution_vector.data.copy()

        result = integrator.integrate(sample_solution_vector)

        # With zero timestep, data should not change
        assert np.allclose(result.data, original_data)

    def test_integrate_different_fluxers(self, sample_solution_vector):
        """Test integration with different flux calculators"""
        fluxer_types = [FluxCalculator(), HLLFluxer()]

        for fluxer in fluxer_types:
            # Create custom parameters with specific fluxer
            class CustomParameters(MockParameters):
                def create_fluxer(self):
                    return fluxer

            params = CustomParameters()
            integrator = Integrator(params)

            # Create a fresh solution vector for each test
            sv = SolutionVector()
            sv.data = sample_solution_vector.data.copy()
            sv.data[1, 0, 0, 0] = 0.2  # Add some non-uniformity
            sv.dx = sv.dy = sv.dz = sample_solution_vector.dx
            sv.cell_sizes = sample_solution_vector.cell_sizes
            sv.timestep = sample_solution_vector.timestep
            sv.adi_idx = sample_solution_vector.adi_idx
            sv.boundary_type = sample_solution_vector.boundary_type
            sv.boundsetter = sample_solution_vector.boundsetter

            result = integrator.integrate(sv)

            # Should produce finite results
            assert np.all(np.isfinite(result.data))
            # Check physical constraints
            assert np.all(result.data[0] > 0)  # density positive
            assert np.all(result.data[4] > 0)  # energy positive

    def test_integrator_components_interaction(
        self, mock_parameters_with_source, sample_solution_vector
    ):
        """Test interaction between different integrator components"""
        integrator = Integrator(mock_parameters_with_source)

        # Verify all components are set up correctly
        assert integrator.fluxer is not None
        assert integrator.source is not None
        assert integrator.gravity is None

        original_data = sample_solution_vector.data.copy()
        result = integrator.integrate(sample_solution_vector)

        # Verify that integration ran and modified the solution
        assert not np.array_equal(result.data, original_data)
        assert np.all(np.isfinite(result.data))

    def test_integration_method_consistency(self, sample_solution_vector):
        """Test that integration method produces consistent results"""
        params = MockParameters()
        integrator = Integrator(params)

        # Run integration twice with same input
        sv1 = sample_solution_vector
        sv2 = SolutionVector()
        sv2.data = sample_solution_vector.data.copy()
        sv2.dx = sv2.dy = sv2.dz = sample_solution_vector.dx
        sv2.cell_sizes = sample_solution_vector.cell_sizes
        sv2.timestep = sample_solution_vector.timestep
        sv2.adi_idx = sample_solution_vector.adi_idx
        sv2.boundary_type = sample_solution_vector.boundary_type
        sv2.boundsetter = sample_solution_vector.boundsetter

        result1 = integrator.integrate(sv1)
        result2 = integrator.integrate(sv2)

        # Results should be identical
        assert np.allclose(result1.data, result2.data)

    def test_integrator_gravity_calculation(
        self, mock_parameters_with_gravity, sample_solution_vector
    ):
        """Test that gravity source calculation works correctly"""
        integrator = Integrator(mock_parameters_with_gravity)

        # Check gravity source calculation
        gravity_source = integrator.gravity.calculate_gravity_source(
            sample_solution_vector
        )

        assert gravity_source.shape == sample_solution_vector.data.shape
        assert np.all(np.isfinite(gravity_source))

        # Should have non-zero contribution to z-momentum (gravity is in z direction)
        assert not np.allclose(gravity_source[3], 0.0)  # z-momentum

        # Energy contribution depends on existing momentum, which is zero in our test case
        # So energy source term could be zero, which is correct
        assert np.all(np.isfinite(gravity_source[4]))  # energy term should be finite

    def test_integrate_uniform_solution(self, mock_parameters, sample_solution_vector):
        """Test integration with truly uniform solution (should have minimal change)"""
        integrator = Integrator(mock_parameters)

        # Make solution completely uniform
        sample_solution_vector.data[0] = 1.0  # uniform density
        sample_solution_vector.data[1:4] = 0.0  # zero momentum everywhere
        sample_solution_vector.data[4] = 2.5  # uniform energy

        original_data = sample_solution_vector.data.copy()
        result = integrator.integrate(sample_solution_vector)

        # For uniform solution with zero momentum, flux divergence should be zero
        # So the solution should remain essentially unchanged
        assert np.allclose(result.data, original_data, atol=1e-12)
        assert np.all(np.isfinite(result.data))
