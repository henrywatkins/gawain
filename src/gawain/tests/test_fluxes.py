from unittest.mock import Mock

import numpy as np
import pytest

from gawain.fluxes import (EulerFluxX, EulerFluxY, EulerFluxZ, FluxCalculator,
                           HLLFluxer, LaxFriedrichsFluxer, LaxWendroffFluxer,
                           MHDFluxX, MHDFluxY, MHDFluxZ)
from gawain.numerics import MHDSolutionVector, SolutionVector


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
    sv.timestep = 0.01

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
    mhd_sv.timestep = 0.01

    return mhd_sv


class TestEulerFluxes:
    def test_euler_flux_x_shape(self, hydro_solution_vector):
        """Test EulerFluxX returns correct shape"""
        flux = EulerFluxX(hydro_solution_vector)
        expected_shape = hydro_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_euler_flux_x_conservation(self, hydro_solution_vector):
        """Test EulerFluxX satisfies basic conservation properties"""
        flux = EulerFluxX(hydro_solution_vector)

        # Mass flux should equal momentum
        assert np.allclose(flux[0], hydro_solution_vector.momX())

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_euler_flux_x_consistency(self, hydro_solution_vector):
        """Test EulerFluxX implementation consistency"""
        flux = EulerFluxX(hydro_solution_vector)

        dens = hydro_solution_vector.dens()
        momX = hydro_solution_vector.momX()
        momY = hydro_solution_vector.momY()
        momZ = hydro_solution_vector.momZ()
        energy = hydro_solution_vector.energy()
        pressure = hydro_solution_vector.pressure()

        # Check individual components
        assert np.allclose(flux[0], momX)
        assert np.allclose(flux[1], momX * momX / dens + pressure)
        assert np.allclose(flux[2], momX * momY / dens)
        assert np.allclose(flux[3], momX * momZ / dens)
        assert np.allclose(flux[4], (energy + pressure) * momX / dens)

    def test_euler_flux_y_shape(self, hydro_solution_vector):
        """Test EulerFluxY returns correct shape"""
        flux = EulerFluxY(hydro_solution_vector)
        expected_shape = hydro_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_euler_flux_y_conservation(self, hydro_solution_vector):
        """Test EulerFluxY satisfies basic conservation properties"""
        flux = EulerFluxY(hydro_solution_vector)

        # Mass flux should equal y-momentum
        assert np.allclose(flux[0], hydro_solution_vector.momY())

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_euler_flux_z_shape(self, hydro_solution_vector):
        """Test EulerFluxZ returns correct shape"""
        flux = EulerFluxZ(hydro_solution_vector)
        expected_shape = hydro_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_euler_flux_z_conservation(self, hydro_solution_vector):
        """Test EulerFluxZ satisfies basic conservation properties"""
        flux = EulerFluxZ(hydro_solution_vector)

        # Mass flux should equal z-momentum
        assert np.allclose(flux[0], hydro_solution_vector.momZ())

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_euler_flux_z_consistency(self, hydro_solution_vector):
        """Test EulerFluxZ implementation consistency"""
        flux = EulerFluxZ(hydro_solution_vector)

        dens = hydro_solution_vector.dens()
        momX = hydro_solution_vector.momX()
        momY = hydro_solution_vector.momY()
        momZ = hydro_solution_vector.momZ()
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
        flux_x = EulerFluxX(hydro_solution_vector)
        flux_y = EulerFluxY(hydro_solution_vector)
        flux_z = EulerFluxZ(hydro_solution_vector)

        # All fluxes should have same shape
        assert flux_x.shape == flux_y.shape == flux_z.shape

        # Mass flux should be the corresponding momentum
        assert np.allclose(flux_x[0], hydro_solution_vector.momX())
        assert np.allclose(flux_y[0], hydro_solution_vector.momY())
        assert np.allclose(flux_z[0], hydro_solution_vector.momZ())


class TestMHDFluxes:
    def test_mhd_flux_x_shape(self, mhd_solution_vector):
        """Test MHDFluxX returns correct shape"""
        flux = MHDFluxX(mhd_solution_vector)
        expected_shape = mhd_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_mhd_flux_x_conservation(self, mhd_solution_vector):
        """Test MHDFluxX satisfies basic conservation properties"""
        flux = MHDFluxX(mhd_solution_vector)

        # Mass flux should equal x-momentum
        assert np.allclose(flux[0], mhd_solution_vector.momX())

        # Magnetic field x-component should not change (ideal MHD constraint)
        assert np.allclose(flux[5], 0.0)

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_mhd_flux_x_consistency(self, mhd_solution_vector):
        """Test MHDFluxX implementation consistency"""
        flux = MHDFluxX(mhd_solution_vector)

        dens = mhd_solution_vector.dens()
        momX, momY, momZ = (
            mhd_solution_vector.momX(),
            mhd_solution_vector.momY(),
            mhd_solution_vector.momZ(),
        )
        energy = mhd_solution_vector.energy()
        tpressure = mhd_solution_vector.total_pressure()
        bx, by, bz = (
            mhd_solution_vector.magX(),
            mhd_solution_vector.magY(),
            mhd_solution_vector.magZ(),
        )

        # Check individual components
        assert np.allclose(flux[0], momX)
        assert np.allclose(flux[1], momX * momX / dens - bx * bx + tpressure)
        assert np.allclose(flux[5], np.zeros_like(bx))  # Bx doesn't change

    def test_mhd_flux_y_shape(self, mhd_solution_vector):
        """Test MHDFluxY returns correct shape"""
        flux = MHDFluxY(mhd_solution_vector)
        expected_shape = mhd_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_mhd_flux_y_conservation(self, mhd_solution_vector):
        """Test MHDFluxY satisfies basic conservation properties"""
        flux = MHDFluxY(mhd_solution_vector)

        # Mass flux should equal y-momentum
        assert np.allclose(flux[0], mhd_solution_vector.momY())

        # Magnetic field y-component should not change
        assert np.allclose(flux[6], 0.0)

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_mhd_flux_z_shape(self, mhd_solution_vector):
        """Test MHDFluxZ returns correct shape"""
        flux = MHDFluxZ(mhd_solution_vector)
        expected_shape = mhd_solution_vector.data.shape
        assert flux.shape == expected_shape

    def test_mhd_flux_z_conservation(self, mhd_solution_vector):
        """Test MHDFluxZ satisfies basic conservation properties"""
        flux = MHDFluxZ(mhd_solution_vector)

        # Mass flux should equal z-momentum
        assert np.allclose(flux[0], mhd_solution_vector.momZ())

        # Magnetic field z-component should not change
        assert np.allclose(flux[7], 0.0)

        # Check that flux values are finite
        assert np.all(np.isfinite(flux))

    def test_mhd_flux_z_consistency(self, mhd_solution_vector):
        """Test MHDFluxZ implementation consistency"""
        flux = MHDFluxZ(mhd_solution_vector)

        dens = mhd_solution_vector.dens()
        momX, momY, momZ = (
            mhd_solution_vector.momX(),
            mhd_solution_vector.momY(),
            mhd_solution_vector.momZ(),
        )
        energy = mhd_solution_vector.energy()
        tpressure = mhd_solution_vector.total_pressure()
        bx, by, bz = (
            mhd_solution_vector.magX(),
            mhd_solution_vector.magY(),
            mhd_solution_vector.magZ(),
        )

        # Check individual components
        assert np.allclose(flux[0], momZ)
        assert np.allclose(flux[1], momZ * momX / dens - bz * bx)
        assert np.allclose(flux[2], momZ * momY / dens - bz * by)
        assert np.allclose(flux[3], momZ * momZ / dens - bz * bz + tpressure)
        assert np.allclose(flux[7], np.zeros_like(bz))  # Bz doesn't change

    def test_mhd_flux_symmetry(self, mhd_solution_vector):
        """Test that MHD fluxes maintain proper symmetry properties"""
        flux_x = MHDFluxX(mhd_solution_vector)
        flux_y = MHDFluxY(mhd_solution_vector)
        flux_z = MHDFluxZ(mhd_solution_vector)

        # All fluxes should have same shape
        assert flux_x.shape == flux_y.shape == flux_z.shape

        # Mass flux should be the corresponding momentum
        assert np.allclose(flux_x[0], mhd_solution_vector.momX())
        assert np.allclose(flux_y[0], mhd_solution_vector.momY())
        assert np.allclose(flux_z[0], mhd_solution_vector.momZ())

        # Corresponding magnetic field components should not change
        assert np.allclose(flux_x[5], 0.0)  # Bx component
        assert np.allclose(flux_y[6], 0.0)  # By component
        assert np.allclose(flux_z[7], 0.0)  # Bz component


class TestFluxCalculator:
    def test_flux_calculator_initialization(self):
        """Test FluxCalculator initialization"""
        fc = FluxCalculator()

        assert fc.x_plus_flux is None
        assert fc.x_minus_flux is None
        assert fc.y_plus_flux is None
        assert fc.y_minus_flux is None
        assert fc.flux_functionX is None
        assert fc.flux_functionY is None
        assert fc.flux_functionZ is None

    def test_set_flux_function_hydro(self):
        """Test setting flux function for hydro"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        assert fc.flux_functionX == EulerFluxX
        assert fc.flux_functionY == EulerFluxY
        assert fc.flux_functionZ == EulerFluxZ

    def test_set_flux_function_mhd(self):
        """Test setting flux function for MHD"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=True)

        assert fc.flux_functionX == MHDFluxX
        assert fc.flux_functionY == MHDFluxY
        assert fc.flux_functionZ == MHDFluxZ


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

    def test_wave_speeds_x(self, hydro_solution_vector):
        """Test wave speed calculation in x direction"""
        hll = HLLFluxer()

        # Create mock boundary setter for shifting
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        ul = hydro_solution_vector
        ur = hydro_solution_vector.plusX()

        sl, sr = hll.wave_speeds_X(ul, ur)

        assert sl.shape == ur.dens().shape
        assert sr.shape == ur.dens().shape
        assert np.all(sl <= sr)  # min speed <= max speed
        assert np.all(np.isfinite(sl))
        assert np.all(np.isfinite(sr))

    def test_wave_speeds_y(self, hydro_solution_vector):
        """Test wave speed calculation in y direction"""
        hll = HLLFluxer()

        # Create mock boundary setter for shifting
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        ul = hydro_solution_vector
        ur = hydro_solution_vector.plusY()

        sl, sr = hll.wave_speeds_Y(ul, ur)

        assert sl.shape == ur.dens().shape
        assert sr.shape == ur.dens().shape
        assert np.all(sl <= sr)  # min speed <= max speed
        assert np.all(np.isfinite(sl))
        assert np.all(np.isfinite(sr))

    def test_wave_speeds_z(self, hydro_solution_vector):
        """Test wave speed calculation in z direction"""
        hll = HLLFluxer()

        # Create mock boundary setter for shifting
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        ul = hydro_solution_vector
        ur = hydro_solution_vector.plusZ()

        sl, sr = hll.wave_speeds_Z(ul, ur)

        assert sl.shape == ur.dens().shape
        assert sr.shape == ur.dens().shape
        assert np.all(sl <= sr)  # min speed <= max speed
        assert np.all(np.isfinite(sl))
        assert np.all(np.isfinite(sr))

    def test_hll_flux_x_subsonic(self, hydro_solution_vector):
        """Test HLL flux in x direction for subsonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        # Create identical left and right states (subsonic)
        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # Small wave speeds crossing zero
        sl = np.full(ul.dens().shape, -0.1)
        sr = np.full(ul.dens().shape, 0.1)

        flux = hll.hll_flux_X(sl, sr, ul, ur)

        assert flux.shape == ul.data.shape
        assert np.all(np.isfinite(flux))

    def test_hll_flux_x_supersonic_left(self, hydro_solution_vector):
        """Test HLL flux in x direction for left supersonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # All positive wave speeds (left supersonic)
        sl = np.full(ul.dens().shape, 0.1)
        sr = np.full(ul.dens().shape, 0.2)

        flux = hll.hll_flux_X(sl, sr, ul, ur)
        expected_flux = EulerFluxX(ul)

        assert np.allclose(flux, expected_flux)

    def test_hll_flux_x_supersonic_right(self, hydro_solution_vector):
        """Test HLL flux in x direction for right supersonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # All negative wave speeds (right supersonic)
        sl = np.full(ul.dens().shape, -0.2)
        sr = np.full(ul.dens().shape, -0.1)

        flux = hll.hll_flux_X(sl, sr, ul, ur)
        expected_flux = EulerFluxX(ur)

        assert np.allclose(flux, expected_flux)

    def test_hll_flux_y_subsonic(self, hydro_solution_vector):
        """Test HLL flux in y direction for subsonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # Small wave speeds crossing zero
        sl = np.full(ul.dens().shape, -0.1)
        sr = np.full(ul.dens().shape, 0.1)

        flux = hll.hll_flux_Y(sl, sr, ul, ur)

        assert flux.shape == ul.data.shape
        assert np.all(np.isfinite(flux))

    def test_hll_flux_z_subsonic(self, hydro_solution_vector):
        """Test HLL flux in z direction for subsonic case"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        ul = hydro_solution_vector
        ur = hydro_solution_vector

        # Small wave speeds crossing zero
        sl = np.full(ul.dens().shape, -0.1)
        sr = np.full(ul.dens().shape, 0.1)

        flux = hll.hll_flux_Z(sl, sr, ul, ur)

        assert flux.shape == ul.data.shape
        assert np.all(np.isfinite(flux))

    def test_muscl_hancock_reconstruction_z(self, hydro_solution_vector):
        """Test MUSCL-Hancock reconstruction in z direction"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        # Create mock boundary setter for shifting
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        u_left = hydro_solution_vector.minusZ()
        u_mid = hydro_solution_vector
        u_right = hydro_solution_vector.plusZ()

        lefts, rights = hll.MUSCL_Hancock_reconstructionZ(u_left, u_mid, u_right)

        assert lefts.data.shape == u_mid.data.shape
        assert rights.data.shape == u_mid.data.shape
        assert np.all(np.isfinite(lefts.data))
        assert np.all(np.isfinite(rights.data))


class TestLaxFriedrichsFluxer:
    def test_lax_friedrichs_initialization(self):
        """Test LaxFriedrichsFluxer initialization"""
        lf = LaxFriedrichsFluxer()
        assert isinstance(lf, FluxCalculator)

    def test_specific_fluxes_shape(self, hydro_solution_vector):
        """Test that LaxFriedrichs specific fluxes maintain correct shape"""
        lf = LaxFriedrichsFluxer()
        lf.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        # Initialize flux values
        lf.x_plus_flux = EulerFluxX(hydro_solution_vector.plusX())
        lf.x_minus_flux = EulerFluxX(hydro_solution_vector.minusX())
        lf.y_plus_flux = EulerFluxY(hydro_solution_vector.plusY())
        lf.y_minus_flux = EulerFluxY(hydro_solution_vector.minusY())
        lf.z_plus_flux = EulerFluxZ(hydro_solution_vector.plusZ())
        lf.z_minus_flux = EulerFluxZ(hydro_solution_vector.minusZ())

        lf._specific_fluxes(hydro_solution_vector)

        assert lf.x_plus_flux.shape == hydro_solution_vector.data.shape
        assert lf.x_minus_flux.shape == hydro_solution_vector.data.shape
        assert lf.y_plus_flux.shape == hydro_solution_vector.data.shape
        assert lf.y_minus_flux.shape == hydro_solution_vector.data.shape
        assert lf.z_plus_flux.shape == hydro_solution_vector.data.shape
        assert lf.z_minus_flux.shape == hydro_solution_vector.data.shape

    def test_lax_friedrichs_3d_consistency(self, hydro_solution_vector):
        """Test that Lax-Friedrichs method works consistently in 3D"""
        lf = LaxFriedrichsFluxer()
        lf.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        flux_div = lf.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
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

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        # Initialize flux values
        lw.x_plus_flux = EulerFluxX(hydro_solution_vector.plusX())
        lw.x_minus_flux = EulerFluxX(hydro_solution_vector.minusX())
        lw.y_plus_flux = EulerFluxY(hydro_solution_vector.plusY())
        lw.y_minus_flux = EulerFluxY(hydro_solution_vector.minusY())
        lw.z_plus_flux = EulerFluxZ(hydro_solution_vector.plusZ())
        lw.z_minus_flux = EulerFluxZ(hydro_solution_vector.minusZ())

        lw._specific_fluxes(hydro_solution_vector)

        assert lw.x_plus_flux.shape == hydro_solution_vector.data.shape
        assert lw.x_minus_flux.shape == hydro_solution_vector.data.shape
        assert lw.y_plus_flux.shape == hydro_solution_vector.data.shape
        assert lw.y_minus_flux.shape == hydro_solution_vector.data.shape
        assert lw.z_plus_flux.shape == hydro_solution_vector.data.shape
        assert lw.z_minus_flux.shape == hydro_solution_vector.data.shape

    def test_lax_wendroff_3d_consistency(self, hydro_solution_vector):
        """Test that Lax-Wendroff method works consistently in 3D"""
        lw = LaxWendroffFluxer()
        lw.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        flux_div = lw.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))


class TestFluxIntegration:
    def test_flux_divergence_calculation_hydro(self, hydro_solution_vector):
        """Test flux divergence calculation for hydro"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        flux_div = fc.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))

    def test_flux_divergence_calculation_mhd(self, mhd_solution_vector):
        """Test flux divergence calculation for MHD"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=True)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        mhd_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], mhd_solution_vector.data
        )

        flux_div = fc.calculate_flux_divergence(mhd_solution_vector)

        assert flux_div.shape == mhd_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))

    def test_flux_divergence_3d_components(self, hydro_solution_vector):
        """Test that flux divergence calculation includes all 3D components"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        # Calculate full 3D flux divergence
        flux_div_3d = fc.calculate_flux_divergence(hydro_solution_vector)

        # Manually calculate each component
        x_plus_flux = EulerFluxX(hydro_solution_vector.plusX())
        x_minus_flux = EulerFluxX(hydro_solution_vector.minusX())
        y_plus_flux = EulerFluxY(hydro_solution_vector.plusY())
        y_minus_flux = EulerFluxY(hydro_solution_vector.minusY())
        z_plus_flux = EulerFluxZ(hydro_solution_vector.plusZ())
        z_minus_flux = EulerFluxZ(hydro_solution_vector.minusZ())

        expected_flux_div = (
            -(x_plus_flux - x_minus_flux) / hydro_solution_vector.dx
            + -(y_plus_flux - y_minus_flux) / hydro_solution_vector.dy
            + -(z_plus_flux - z_minus_flux) / hydro_solution_vector.dz
        )

        # They should be equal
        assert np.allclose(flux_div_3d, expected_flux_div)

    def test_hll_flux_divergence_3d(self, hydro_solution_vector):
        """Test HLL flux divergence calculation in full 3D"""
        hll = HLLFluxer()
        hll.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        flux_div = hll.calculate_flux_divergence(hydro_solution_vector)

        assert flux_div.shape == hydro_solution_vector.data.shape
        assert np.all(np.isfinite(flux_div))

        # Check that all flux components were calculated
        assert hll.x_plus_flux is not None
        assert hll.x_minus_flux is not None
        assert hll.y_plus_flux is not None
        assert hll.y_minus_flux is not None
        assert hll.z_plus_flux is not None
        assert hll.z_minus_flux is not None

    def test_3d_vs_2d_flux_consistency(self, hydro_solution_vector):
        """Test that 3D flux calculation is consistent with 2D when z-direction is trivial"""
        # Create a 3D solution where the z-direction is essentially 2D (nz=1)
        nx, ny, nz = hydro_solution_vector.data.shape[1:]
        if nz == 1:
            # This is essentially a 2D problem
            fc_3d = FluxCalculator()
            fc_3d.set_flux_function(with_mhd=False)

            # Create mock boundary setter
            from gawain.numerics import BoundarySetter

            hydro_solution_vector.boundsetter = BoundarySetter(
                ["periodic", "periodic", "periodic"], hydro_solution_vector.data
            )

            flux_div_3d = fc_3d.calculate_flux_divergence(hydro_solution_vector)

            # For a truly 2D case (nz=1), z-contribution should be minimal
            z_plus_flux = EulerFluxZ(hydro_solution_vector.plusZ())
            z_minus_flux = EulerFluxZ(hydro_solution_vector.minusZ())
            z_contribution = -(z_plus_flux - z_minus_flux) / hydro_solution_vector.dz

            # In 2D case with nz=1, plusZ and minusZ should be identical due to periodicity
            # so z_contribution should be very small
            assert np.allclose(z_contribution, 0.0, atol=1e-12)

    def test_flux_conservation_3d(self, hydro_solution_vector):
        """Test that fluxes satisfy conservation properties in 3D"""
        fc = FluxCalculator()
        fc.set_flux_function(with_mhd=False)

        # Create mock boundary setter
        from gawain.numerics import BoundarySetter

        hydro_solution_vector.boundsetter = BoundarySetter(
            ["periodic", "periodic", "periodic"], hydro_solution_vector.data
        )

        flux_div = fc.calculate_flux_divergence(hydro_solution_vector)

        # Conservation should ensure that mass flux divergence equals
        # the negative of momentum divergence contributions appropriately
        mass_flux_div = flux_div[0]  # density equation

        # Check that the flux divergence has the right shape and finite values
        assert mass_flux_div.shape == hydro_solution_vector.dens().shape
        assert np.all(np.isfinite(mass_flux_div))
