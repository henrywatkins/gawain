import numpy as np
import pytest

from gawain.fluxes import (EulerFlux, FluxCalculator, HLLFluxer,
                           LaxFriedrichsFluxer, LaxWendroffFluxer, MHDFlux)
from gawain.numerics import BoundarySetter, MHDSolutionVector, SolutionVector


@pytest.fixture
def hydro_solution_vector():
    """Create a hydro solution vector for testing"""
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
def mhd_solution_vector():
    """Create an MHD solution vector for testing"""
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


class TestEulerFluxes:
    def test_euler_flux_x_shape(self, hydro_solution_vector):
        """Test EulerFlux in X direction returns correct shape"""
        flux = EulerFlux(hydro_solution_vector, 0)
        expected_shape = hydro_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_euler_flux_x_conservation(self, hydro_solution_vector):
        """Test EulerFlux in X direction satisfies basic conservation properties"""
        flux = EulerFlux(hydro_solution_vector, 0)

        # Mass flux should equal x-momentum
        assert np.allclose(flux[0], hydro_solution_vector.mom(0))

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_euler_flux_x_consistency(self, hydro_solution_vector):
        """Test EulerFlux in X direction implementation consistency"""
        flux = EulerFlux(hydro_solution_vector, 0)

        dens = hydro_solution_vector.dens()
        momX = hydro_solution_vector.mom(0)
        momY = hydro_solution_vector.mom(1)
        momZ = hydro_solution_vector.mom(2)
        energy = hydro_solution_vector.energy()
        pressure = hydro_solution_vector.pressure()

        # Check individual components
        assert np.allclose(flux[0], momX)
        assert np.allclose(flux[1], momX * momX / dens + pressure)
        assert np.allclose(flux[2], momX * momY / dens)
        assert np.allclose(flux[3], momX * momZ / dens)
        assert np.allclose(flux[4], (energy + pressure) * momX / dens)

    def test_euler_flux_y_shape(self, hydro_solution_vector):
        """Test EulerFlux in Y direction returns correct shape"""
        flux = EulerFlux(hydro_solution_vector, 1)
        expected_shape = hydro_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_euler_flux_y_conservation(self, hydro_solution_vector):
        """Test EulerFlux in Y direction satisfies basic conservation properties"""
        flux = EulerFlux(hydro_solution_vector, 1)

        # Mass flux should equal y-momentum
        assert np.allclose(flux[0], hydro_solution_vector.mom(1))

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_euler_flux_y_consistency(self, hydro_solution_vector):
        """Test EulerFlux in Y direction implementation consistency"""
        flux = EulerFlux(hydro_solution_vector, 1)

        dens = hydro_solution_vector.dens()
        momX = hydro_solution_vector.mom(0)
        momY = hydro_solution_vector.mom(1)
        momZ = hydro_solution_vector.mom(2)
        energy = hydro_solution_vector.energy()
        pressure = hydro_solution_vector.pressure()

        # Check individual components
        assert np.allclose(flux[0], momY)
        assert np.allclose(flux[1], momY * momX / dens)
        assert np.allclose(flux[2], momY * momY / dens + pressure)
        assert np.allclose(flux[3], momY * momZ / dens)
        assert np.allclose(flux[4], (energy + pressure) * momY / dens)

    def test_euler_flux_z_shape(self, hydro_solution_vector):
        """Test EulerFlux in Z direction returns correct shape"""
        flux = EulerFlux(hydro_solution_vector, 2)
        expected_shape = hydro_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_euler_flux_z_conservation(self, hydro_solution_vector):
        """Test EulerFlux in Z direction satisfies basic conservation properties"""
        flux = EulerFlux(hydro_solution_vector, 2)

        # Mass flux should equal z-momentum
        assert np.allclose(flux[0], hydro_solution_vector.mom(2))

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_euler_flux_z_consistency(self, hydro_solution_vector):
        """Test EulerFlux in Z direction implementation consistency"""
        flux = EulerFlux(hydro_solution_vector, 2)

        dens = hydro_solution_vector.dens()
        momX = hydro_solution_vector.mom(0)
        momY = hydro_solution_vector.mom(1)
        momZ = hydro_solution_vector.mom(2)
        energy = hydro_solution_vector.energy()
        pressure = hydro_solution_vector.pressure()

        # Check individual components
        assert np.allclose(flux[0], momZ)
        assert np.allclose(flux[1], momZ * momX / dens)
        assert np.allclose(flux[2], momZ * momY / dens)
        assert np.allclose(flux[3], momZ * momZ / dens + pressure)
        assert np.allclose(flux[4], (energy + pressure) * momZ / dens)

    def test_euler_flux_symmetry(self, hydro_solution_vector):
        """Test that Euler fluxes maintain proper symmetry properties"""
        flux_x = EulerFlux(hydro_solution_vector, 0)
        flux_y = EulerFlux(hydro_solution_vector, 1)
        flux_z = EulerFlux(hydro_solution_vector, 2)

        # All fluxes should have same shape
        assert flux_x.shape == flux_y.shape == flux_z.shape

        # Mass flux should be the corresponding momentum
        assert np.allclose(flux_x[0], hydro_solution_vector.mom(0))
        assert np.allclose(flux_y[0], hydro_solution_vector.mom(1))
        assert np.allclose(flux_z[0], hydro_solution_vector.mom(2))

    def test_euler_flux_all_axes(self, hydro_solution_vector):
        """Test that EulerFlux works for all three axes"""
        for axis in range(3):
            flux = EulerFlux(hydro_solution_vector, axis)
            assert flux.shape == hydro_solution_vector.data.shape
            assert np.all(np.isfinite(flux))
            # Mass flux should equal axis momentum
            assert np.allclose(flux[0], hydro_solution_vector.mom(axis))


class TestMHDFluxes:
    def test_mhd_flux_x_shape(self, mhd_solution_vector):
        """Test MHDFlux in X direction returns correct shape"""
        flux = MHDFlux(mhd_solution_vector, 0)
        expected_shape = mhd_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_mhd_flux_x_conservation(self, mhd_solution_vector):
        """Test MHDFlux in X direction satisfies basic conservation properties"""
        flux = MHDFlux(mhd_solution_vector, 0)

        # Mass flux should equal x-momentum
        assert np.allclose(flux[0], mhd_solution_vector.mom(0))

        # Magnetic field x-component should not change (ideal MHD constraint)
        assert np.allclose(flux[5], 0.0)

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_mhd_flux_x_consistency(self, mhd_solution_vector):
        """Test MHDFlux in X direction implementation consistency"""
        flux = MHDFlux(mhd_solution_vector, 0)

        dens = mhd_solution_vector.dens()
        momX, momY, momZ = (
            mhd_solution_vector.mom(0),
            mhd_solution_vector.mom(1),
            mhd_solution_vector.mom(2),
        )
        energy = mhd_solution_vector.energy()
        tpressure = mhd_solution_vector.total_pressure()
        bx, by, bz = (
            mhd_solution_vector.mag(0),
            mhd_solution_vector.mag(1),
            mhd_solution_vector.mag(2),
        )

        # Check individual components
        assert np.allclose(flux[0], momX)
        assert np.allclose(flux[1], momX * momX / dens - bx * bx + tpressure)
        assert np.allclose(flux[5], np.zeros_like(bx))  # Bx doesn't change

    def test_mhd_flux_y_shape(self, mhd_solution_vector):
        """Test MHDFlux in Y direction returns correct shape"""
        flux = MHDFlux(mhd_solution_vector, 1)
        expected_shape = mhd_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_mhd_flux_y_conservation(self, mhd_solution_vector):
        """Test MHDFlux in Y direction satisfies basic conservation properties"""
        flux = MHDFlux(mhd_solution_vector, 1)

        # Mass flux should equal y-momentum
        assert np.allclose(flux[0], mhd_solution_vector.mom(1))

        # Magnetic field y-component should not change
        assert np.allclose(flux[6], 0.0)

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_mhd_flux_y_consistency(self, mhd_solution_vector):
        """Test MHDFlux in Y direction implementation consistency"""
        flux = MHDFlux(mhd_solution_vector, 1)

        dens = mhd_solution_vector.dens()
        momX, momY, momZ = (
            mhd_solution_vector.mom(0),
            mhd_solution_vector.mom(1),
            mhd_solution_vector.mom(2),
        )
        energy = mhd_solution_vector.energy()
        tpressure = mhd_solution_vector.total_pressure()
        bx, by, bz = (
            mhd_solution_vector.mag(0),
            mhd_solution_vector.mag(1),
            mhd_solution_vector.mag(2),
        )

        # Check individual components
        assert np.allclose(flux[0], momY)
        assert np.allclose(flux[2], momY * momY / dens - by * by + tpressure)
        assert np.allclose(flux[6], np.zeros_like(by))  # By doesn't change

    def test_mhd_flux_z_shape(self, mhd_solution_vector):
        """Test MHDFlux in Z direction returns correct shape"""
        flux = MHDFlux(mhd_solution_vector, 2)
        expected_shape = mhd_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_mhd_flux_z_conservation(self, mhd_solution_vector):
        """Test MHDFlux in Z direction satisfies basic conservation properties"""
        flux = MHDFlux(mhd_solution_vector, 2)

        # Mass flux should equal z-momentum
        assert np.allclose(flux[0], mhd_solution_vector.mom(2))

        # Magnetic field z-component should not change
        assert np.allclose(flux[7], 0.0)

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_mhd_flux_z_consistency(self, mhd_solution_vector):
        """Test MHDFlux in Z direction implementation consistency"""
        flux = MHDFlux(mhd_solution_vector, 2)

        dens = mhd_solution_vector.dens()
        momX, momY, momZ = (
            mhd_solution_vector.mom(0),
            mhd_solution_vector.mom(1),
            mhd_solution_vector.mom(2),
        )
        energy = mhd_solution_vector.energy()
        tpressure = mhd_solution_vector.total_pressure()
        bx, by, bz = (
            mhd_solution_vector.mag(0),
            mhd_solution_vector.mag(1),
            mhd_solution_vector.mag(2),
        )

        # Check individual components
        assert np.allclose(flux[0], momZ)
        assert np.allclose(flux[1], momZ * momX / dens - bz * bx)
        assert np.allclose(flux[2], momZ * momY / dens - bz * by)
        assert np.allclose(flux[3], momZ * momZ / dens - bz * bz + tpressure)
        assert np.allclose(flux[7], np.zeros_like(bz))  # Bz doesn't change

    def test_mhd_flux_symmetry(self, mhd_solution_vector):
        """Test that MHD fluxes maintain proper symmetry properties"""
        flux_x = MHDFlux(mhd_solution_vector, 0)
        flux_y = MHDFlux(mhd_solution_vector, 1)
        flux_z = MHDFlux(mhd_solution_vector, 2)

        # All fluxes should have same shape
        assert flux_x.shape == flux_y.shape == flux_z.shape

        # Mass flux should be the corresponding momentum
        assert np.allclose(flux_x[0], mhd_solution_vector.mom(0))
        assert np.allclose(flux_y[0], mhd_solution_vector.mom(1))
        assert np.allclose(flux_z[0], mhd_solution_vector.mom(2))

        # Corresponding magnetic field components should not change
        assert np.allclose(flux_x[5], 0.0)  # Bx component
        assert np.allclose(flux_y[6], 0.0)  # By component
        assert np.allclose(flux_z[7], 0.0)  # Bz component

    def test_mhd_flux_all_axes(self, mhd_solution_vector):
        """Test that MHDFlux works for all three axes"""
        for axis in range(3):
            flux = MHDFlux(mhd_solution_vector, axis)
            assert flux.shape == mhd_solution_vector.data.shape
            assert np.all(np.isfinite(flux))
            # Mass flux should equal axis momentum
            assert np.allclose(flux[0], mhd_solution_vector.mom(axis))
            # Normal magnetic field component shouldn't change
            assert np.allclose(flux[axis + 5], 0.0)


class TestFluxCalculator:
    def test_flux_calculator_initialization(self):
        """Test FluxCalculator initialization"""
        fc = FluxCalculator()

        assert fc.flux_function is None

    def test_set_flux_function_hydro(self):
        """Test setting flux function for hydro"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        assert fc.flux_function == EulerFlux

    def test_set_flux_function_mhd(self):
        """Test setting flux function for MHD"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=True)

        assert fc.flux_function == MHDFlux

    def test_specific_fluxes_hydro(self, hydro_solution_vector):
        """Test _specific_fluxes method for hydro"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        fluxes = fc._specific_fluxes(hydro_solution_vector)

        # Should return 6 fluxes (3 axes × 2 directions)
        assert len(fluxes) == 6
        for flux in fluxes:
            assert flux.shape == hydro_solution_vector.data.shape
            assert np.all(np.isfinite(flux))

    def test_specific_fluxes_mhd(self, mhd_solution_vector):
        """Test _specific_fluxes method for MHD"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=True)

        fluxes = fc._specific_fluxes(mhd_solution_vector)

        # Should return 6 fluxes (3 axes × 2 directions)
        assert len(fluxes) == 6
        for flux in fluxes:
            assert flux.shape == mhd_solution_vector.data.shape
            assert np.all(np.isfinite(flux))

    def test_calculate_flux_divergence_hydro(self, hydro_solution_vector):
        """Test flux divergence calculation for hydro"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        flux_div = fc.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))

    def test_calculate_flux_divergence_mhd(self, mhd_solution_vector):
        """Test flux divergence calculation for MHD"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=True)

        flux_div = fc.calculate_flux_divergence(mhd_solution_vector)

        assert flux_div.shape == mhd_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))


class TestHLLFluxer:
    def test_hll_fluxer_initialization(self):
        """Test HLLFluxer initialization"""
        hll = HLLFluxer()
        assert isinstance(hll, FluxCalculator)

    def test_minmod_limiter_positive(self):
        """Test minmod limiter with positive values"""
        hll = HLLFluxer()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 1.0, 4.0])

        result = hll.minmod(a, b)
        expected = np.array([1.0, 1.0, 3.0])  # min of absolute values with same sign

        assert np.allclose(result, expected)

    def test_minmod_limiter_opposite_signs(self):
        """Test minmod limiter with opposite signs"""
        hll = HLLFluxer()
        a = np.array([1.0, -2.0])
        b = np.array([-1.0, 2.0])

        result = hll.minmod(a, b)
        expected = np.array([0.0, 0.0])  # opposite signs should give zero

        assert np.allclose(result, expected)

    def test_minmod_limiter_zero_case(self):
        """Test minmod limiter with zero values"""
        hll = HLLFluxer()
        a = np.array([0.0, 1.0])
        b = np.array([2.0, 0.0])

        result = hll.minmod(a, b)
        expected = np.array([0.0, 0.0])

        assert np.allclose(result, expected)

    def test_wave_speeds(self, hydro_solution_vector):
        """Test wave speed calculation"""
        hll = HLLFluxer()

        ul = hydro_solution_vector
        ur = hydro_solution_vector.get_neighbour_state(0, 1)

        sl, sr = hll.wave_speeds(ul, ur, 0)

        assert sl.shape == ur.dens().shape
        assert sr.shape == ur.dens().shape
        assert np.all(sl <= sr)  # min speed <= max speed
        assert np.all(np.isfinite(sl))
        assert np.all(np.isfinite(sr))

    def test_wave_speeds_all_axes(self, hydro_solution_vector):
        """Test wave speed calculation for all axes"""
        hll = HLLFluxer()

        for axis in range(3):
            ul = hydro_solution_vector
            ur = hydro_solution_vector.get_neighbour_state(axis, 1)

            sl, sr = hll.wave_speeds(ul, ur, axis)

            assert sl.shape == ur.dens().shape
            assert sr.shape == ur.dens().shape
            assert np.all(sl <= sr)
            assert np.all(np.isfinite(sl))
            assert np.all(np.isfinite(sr))

    def test_wave_speeds_mhd(self, mhd_solution_vector):
        """Test wave speed calculation for MHD"""
        hll = HLLFluxer()

        ul = mhd_solution_vector
        ur = mhd_solution_vector.get_neighbour_state(0, 1)

        sl, sr = hll.wave_speeds(ul, ur, 0)

        assert sl.shape == ur.dens().shape
        assert sr.shape == ur.dens().shape
        assert np.all(sl <= sr)
        assert np.all(np.isfinite(sl))
        assert np.all(np.isfinite(sr))

    def test_hll_flux_subsonic(self, hydro_solution_vector):
        """Test HLL flux for subsonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        # Create identical left and right states (subsonic)
        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # Small wave speeds crossing zero
        sl = np.full(ul.dens().shape, -0.1)
        sr = np.full(ul.dens().shape, 0.1)

        flux = hll.hll_flux(sl, sr, ul, ur, 0)

        assert flux.shape == ul.data.shape
        assert np.all(np.isfinite(flux))

    def test_hll_flux_supersonic_left(self, hydro_solution_vector):
        """Test HLL flux for left supersonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # All positive wave speeds (left supersonic)
        sl = np.full(ul.dens().shape, 0.1)
        sr = np.full(ul.dens().shape, 0.2)

        flux = hll.hll_flux(sl, sr, ul, ur, 0)
        expected_flux = EulerFlux(ul, 0)

        assert np.allclose(flux, expected_flux)

    def test_hll_flux_supersonic_right(self, hydro_solution_vector):
        """Test HLL flux for right supersonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # All negative wave speeds (right supersonic)
        sl = np.full(ul.dens().shape, -0.2)
        sr = np.full(ul.dens().shape, -0.1)

        flux = hll.hll_flux(sl, sr, ul, ur, 0)
        expected_flux = EulerFlux(ur, 0)

        assert np.allclose(flux, expected_flux)

    def test_hll_flux_all_axes(self, hydro_solution_vector):
        """Test HLL flux for all axes"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        sl = np.full(ul.dens().shape, -0.1)
        sr = np.full(ul.dens().shape, 0.1)

        for axis in range(3):
            flux = hll.hll_flux(sl, sr, ul, ur, axis)
            assert flux.shape == ul.data.shape
            assert np.all(np.isfinite(flux))

    def test_muscl_hancock_reconstruction(self, hydro_solution_vector):
        """Test MUSCL-Hancock reconstruction"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        u_left = hydro_solution_vector.get_neighbour_state(0, -1)
        u_mid = hydro_solution_vector
        u_right = hydro_solution_vector.get_neighbour_state(0, 1)

        lefts, rights = hll.MUSCL_Hancock_reconstruction(u_left, u_mid, u_right, 0)

        assert lefts[0].data.shape == u_mid.data.shape
        assert lefts[1].data.shape == u_mid.data.shape
        assert rights[0].data.shape == u_mid.data.shape
        assert rights[1].data.shape == u_mid.data.shape
        assert np.all(np.isfinite(lefts[0].data))
        assert np.all(np.isfinite(lefts[1].data))
        assert np.all(np.isfinite(rights[0].data))
        assert np.all(np.isfinite(rights[1].data))

    def test_hll_directional_fluxes(self, hydro_solution_vector):
        """Test HLL directional flux calculation"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        right_flux, left_flux = hll.hll_directional_fluxes(hydro_solution_vector, 0)

        assert right_flux.shape == hydro_solution_vector.data.shape
        assert left_flux.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(right_flux))
        assert np.all(np.isfinite(left_flux))

    def test_hll_flux_divergence(self, hydro_solution_vector):
        """Test complete HLL flux divergence calculation"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        flux_div = hll.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))


class TestLaxFriedrichsFluxer:
    def test_lax_friedrichs_initialization(self):
        """Test LaxFriedrichsFluxer initialization"""
        lf = LaxFriedrichsFluxer()
        assert isinstance(lf, FluxCalculator)

    def test_specific_fluxes_shape(self, hydro_solution_vector):
        """Test that LaxFriedrichs specific fluxes maintain correct shape"""
        lf = LaxFriedrichsFluxer()
        lf.set_flux_function(with_mhd=False)

        fluxes = lf._specific_fluxes(hydro_solution_vector)

        assert len(fluxes) == 6  # 3 axes × 2 directions
        for flux in fluxes:
            assert flux.shape == hydro_solution_vector.data.shape
            assert np.all(np.isfinite(flux))

    def test_lax_friedrichs_flux_divergence(self, hydro_solution_vector):
        """Test that Lax-Friedrichs method works consistently"""
        lf = LaxFriedrichsFluxer()
        lf.set_flux_function(with_mhd=False)

        flux_div = lf.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))

    def test_lax_friedrichs_mhd(self, mhd_solution_vector):
        """Test Lax-Friedrichs with MHD"""
        lf = LaxFriedrichsFluxer()
        lf.set_flux_function(with_mhd=True)

        flux_div = lf.calculate_flux_divergence(mhd_solution_vector)

        assert flux_div.shape == mhd_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))


class TestLaxWendroffFluxer:
    def test_lax_wendroff_initialization(self):
        """Test LaxWendroffFluxer initialization"""
        lw = LaxWendroffFluxer()
        assert isinstance(lw, FluxCalculator)

    def test_specific_fluxes_shape(self, hydro_solution_vector):
        """Test that LaxWendroff specific fluxes maintain correct shape"""
        lw = LaxWendroffFluxer()
        lw.set_flux_function(with_mhd=False)

        fluxes = lw._specific_fluxes(hydro_solution_vector)

        assert len(fluxes) == 6  # 3 axes × 2 directions
        for flux in fluxes:
            assert flux.shape == hydro_solution_vector.data.shape
            assert np.all(np.isfinite(flux))

    def test_lax_wendroff_flux_divergence(self, hydro_solution_vector):
        """Test that Lax-Wendroff method works consistently"""
        lw = LaxWendroffFluxer()
        lw.set_flux_function(with_mhd=False)

        flux_div = lw.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))

    def test_lax_wendroff_mhd(self, mhd_solution_vector):
        """Test Lax-Wendroff with MHD"""
        lw = LaxWendroffFluxer()
        lw.set_flux_function(with_mhd=True)

        flux_div = lw.calculate_flux_divergence(mhd_solution_vector)

        assert flux_div.shape == mhd_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))


class TestFluxIntegration:
    def test_flux_method_comparison(self, hydro_solution_vector):
        """Test that different flux methods give reasonable results"""
        methods = [
            FluxCalculator(),
            HLLFluxer(),
            LaxFriedrichsFluxer(),
            LaxWendroffFluxer(),
        ]

        flux_divs = []
        for method in methods:
            method.set_flux_function(with_mhd=False)
            flux_div = method.calculate_flux_divergence(hydro_solution_vector)
            flux_divs.append(flux_div)

            # All methods should give finite results
            assert np.all(np.isfinite(flux_div))
            assert flux_div.shape == hydro_solution_vector.data.shape

    def test_flux_consistency_across_axes(self, hydro_solution_vector):
        """Test that flux calculations are consistent across all spatial axes"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        # Test that fluxes work in all directions
        for axis in range(3):
            flux = EulerFlux(hydro_solution_vector, axis)
            assert flux.shape == hydro_solution_vector.data.shape
            assert np.all(np.isfinite(flux))
            # Mass flux should equal momentum in the specified axis
            assert np.allclose(flux[0], hydro_solution_vector.mom(axis))

    def test_mhd_flux_consistency_across_axes(self, mhd_solution_vector):
        """Test that MHD flux calculations are consistent across all spatial axes"""
        for axis in range(3):
            flux = MHDFlux(mhd_solution_vector, axis)
            assert flux.shape == mhd_solution_vector.data.shape
            assert np.all(np.isfinite(flux))
            # Mass flux should equal momentum in the specified axis
            assert np.allclose(flux[0], mhd_solution_vector.mom(axis))
            # Normal magnetic field component shouldn't change
            assert np.allclose(flux[axis + 5], 0.0)

    def test_flux_divergence_symmetry(self, hydro_solution_vector):
        """Test flux divergence calculation maintains proper symmetry"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        # Create a uniform solution
        uniform_data = np.ones_like(hydro_solution_vector.data)
        uniform_data[0] = 1.0  # density
        uniform_data[1:4] = 0.0  # zero momentum
        uniform_data[4] = 2.5  # energy

        uniform_sv = SolutionVector()
        uniform_sv.data = uniform_data
        uniform_sv.dx = uniform_sv.dy = uniform_sv.dz = 0.1
        uniform_sv.cell_sizes = (0.1, 0.1, 0.1)
        uniform_sv.adi_idx = 1.4
        uniform_sv.boundary_type = ["periodic", "periodic", "periodic"]
        uniform_sv.boundsetter = BoundarySetter(uniform_sv.boundary_type, uniform_data)

        flux_div = fc.calculate_flux_divergence(uniform_sv)

        # For uniform solution with zero momentum, flux divergence should be very small
        assert np.allclose(flux_div, 0.0, atol=1e-12)

    def test_all_fluxers_with_mhd(self, mhd_solution_vector):
        """Test that all flux calculators work with MHD"""
        methods = [
            FluxCalculator(),
            HLLFluxer(),
            LaxFriedrichsFluxer(),
            LaxWendroffFluxer(),
        ]

        for method in methods:
            method.set_flux_function(with_mhd=True)
            flux_div = method.calculate_flux_divergence(mhd_solution_vector)

            assert flux_div.shape == mhd_solution_vector.data.shape
            assert np.all(np.isfinite(flux_div))
            # Check that all 8 MHD variables are updated
            assert flux_div.shape[0] == 8
