"""Flux calculation routine

These functions and classes are used for the
physics-specific flux calculations for both
hydrodynamic and MHD fluxes

"""

import numpy as np


def EulerFlux(u, axis):
    """Calculate the Euler flux along a given axis from a solution vector"""
    dens = u.dens()
    momX, momY, momZ = u.mom(0), u.mom(1), u.mom(2)
    en = u.energy()
    pressure = u.pressure()
    axis_mom = u.mom(axis)
    flux = np.array(
        [
            axis_mom,
            axis_mom * momX / dens,
            axis_mom * momY / dens,
            axis_mom * momZ / dens,
            (en + pressure) * axis_mom / dens,
        ]
    )
    flux[axis + 1] += pressure
    return flux


def MHDFlux(u, axis):
    """Calculate the MHD flux along a given axis from a solution vector"""
    dens = u.dens()
    momX, momY, momZ = u.mom(0), u.mom(1), u.mom(2)
    axis_mom = u.mom(axis)
    axis_b = u.mag(axis)
    en = u.energy()
    tpressure = u.total_pressure()
    bx, by, bz = u.mag(0), u.mag(1), u.mag(2)
    zeros = np.zeros(dens.shape)
    flux = np.array(
        [
            axis_mom,
            axis_mom * momX / dens - axis_b * bx,
            axis_mom * momY / dens - axis_b * by,
            axis_mom * momZ / dens - axis_b * bz,
            (en + tpressure) * axis_mom / dens
            - (bx * momX + by * momY + bz * momZ) * axis_b / dens,
            (axis_mom * bx - momX * axis_b) / dens,
            (axis_mom * by - momY * axis_b) / dens,
            (axis_mom * bz - momZ * axis_b) / dens,
        ]
    )
    flux[axis + 1] += tpressure
    flux[axis + 5] = zeros
    return flux


class FluxCalculator:
    """Base flux calculation utility

    Base class for classes used to calculate the divergence of
    flux term in the conservation problem.

    Attributes
    ----------
    x/y/z_plus/minus_flux : ndarray
        the numerical fluxes in each direction
    """

    def __init__(self):
        self.flux_function = None

    def set_flux_function(self, with_mhd):
        """set the flux calculation function to euler or mhd"""
        self.flux_function = MHDFlux if with_mhd else EulerFlux

    def _specific_fluxes(self, u):
        return (
            self.flux_function(u.get_neighbour_state(i, j), i)
            for i in range(3)
            for j in (1, -1)
        )

    def calculate_flux_divergence(self, u):
        """Flux divergence calculation"""

        fluxes = self._specific_fluxes(u)
        total_flux = -(fluxes[0] - fluxes[1]) / u.dx
        total_flux += -(fluxes[2] - fluxes[3]) / u.dy
        total_flux += -(fluxes[4] - fluxes[5]) / u.dz
        return total_flux


class HLLFluxer(FluxCalculator):
    """A MUSCL-Hancock HLL solver using minmod limiter

    This class calculates the numerical divergence of the flux
    in the conservation equation problem using the HLLE solver
    with MUSCL-hancock reconstruction and a minmod limiter.
    """

    def __init__(self):
        super(HLLFluxer, self).__init__()

    def minmod(self, a, b):
        """minmod limiter"""
        return np.where(
            a * b <= 0.0, 0.0, np.where(np.absolute(a) < np.absolute(b), a, b)
        )

    def wave_speeds(self, Ul, Ur, axis):
        """Calculate the min/max wave speeds at the x-axis inteface between two cells"""
        lambda_L_min, lambda_L_max = Ul.calculate_min_max_wave_speeds(axis)
        lambda_R_min, lambda_R_max = Ur.calculate_min_max_wave_speeds(axis)

        zeros = np.zeros(lambda_L_min.shape)

        return (
            np.minimum(np.minimum(zeros, lambda_L_min), lambda_R_min),
            np.maximum(np.maximum(zeros, lambda_L_max), lambda_R_max),
        )

    def hll_flux(self, Sl, Sr, Ul, Ur, axis):
        """Calculate the HLLE flux at the interface"""

        fl = self.flux_function(Ul, axis)
        fr = self.flux_function(Ur, axis)

        ur = Ur.centroid()
        ul = Ul.centroid()

        fhll = (Sr * fl - Sl * fr + Sl * Sr * (ur - ul)) / (Sr - Sl)

        return np.where(Sl >= 0.0, fl, np.where(Sr <= 0.0, fr, fhll))

    def MUSCL_Hancock_reconstruction(self, U_left, U_mid, U_right, axis):
        """ "Calculate the MUSCL-hancock reconstruction of cell values

        note: this means lefts holds the states for the left of the i+1/2 face,
        rights holds the states for the right of the i-1/2 face.
        """
        # reconstruct
        umid = U_mid.centroid()
        a = umid - U_left.centroid()
        b = U_right.centroid() - umid

        limited = self.minmod(a, b)

        a = umid - 0.5 * limited
        b = umid + 0.5 * limited

        rights = U_mid.copy()
        lefts = U_mid.copy()

        rights.set_centroid(a)
        lefts.set_centroid(b)

        # evolve
        plus_flux = self.flux_function(rights, axis)
        minus_flux = self.flux_function(lefts, axis)

        dt = U_mid.timestep
        delta = U_mid.cell_sizes[axis]

        a = a + 0.5 * dt * (plus_flux - minus_flux) / delta
        b = b + 0.5 * dt * (plus_flux - minus_flux) / delta

        rights.set_centroid(a)
        lefts.set_centroid(b)

        return lefts, rights

    def hlle_directional_flux(self, u, axis):
        uplus = u.get_neighbour_state(axis, 1)
        uminus = u.get_neighbour_state(axis, -1)
        lefts, rights = self.MUSCL_Hancock_reconstruction(uminus, u, uplus, axis)

        umid = rights
        uminus = lefts.get_neighbour_state(0, -1)
        Sl, Sr = self.wave_speeds(uminus, umid, 0)
        minus_flux = self.hll_flux(Sl, Sr, uminus, umid, 0)

        umid = lefts
        uplus = rights.get_neighbour_state(axis, 1)
        Sl, Sr = self.wave_speeds(umid, uplus, axis)
        plus_flux = self.hll_flux(Sl, Sr, umid, uplus, axis)
        return plus_flux, minus_flux

    def _specific_fluxes(self, u):
        """HLLE solver - specifc flux calculation"""
        fluxes = []
        for i in range(3):
            fluxes.extend(self.hlle_directional_flux(u, i))
        return fluxes


class LaxFriedrichsFluxer(FluxCalculator):
    """Lax-Friedrichs method flux calculator

    This class calculates the numerical divergence of the flux
    in the conservation equation problem using the Lax-Friedrichs method.
    """

    def __init__(self):
        super(LaxFriedrichsFluxer, self).__init__()

    def _specific_fluxes(self, u):
        """Lax-Friedrichs -specific flux calculation"""
        u1 = u.centroid()
        dt = u.timestep
        cell_sizes = u.cell_sizes

        fluxes = []
        for i in range(3):
            mid_flux = self.flux_functionX(u, i)
            u2 = u.get_neighbour_state(0, 1).centroid()
            plus_flux = self.flux_function(u.get_neighbour_state(i, 1), i)
            plus_flux = 0.5 * ((plus_flux + mid_flux) - cell_sizes[i] * (u2 - u1) / dt)
            fluxes.append(plus_flux)
            u2 = u.get_neighbour_state(0, -1).centroid()
            minus_flux = self.flux_function(u.get_neighbour_state(i, -1), i)
            minus_flux = 0.5 * (
                (minus_flux + mid_flux) - cell_sizes[i] * (u1 - u2) / dt
            )
            fluxes.append(minus_flux)
        return fluxes


class LaxWendroffFluxer(FluxCalculator):
    """Lax-Wendroff method flux calculator

    This class calculates the numerical divergence of the flux
    in the conservation equation problem using the Lax-Wendroff method.
    """

    def __init__(self):
        super(LaxWendroffFluxer, self).__init__()

    def _specific_fluxes(self, u):
        """Lax-Wendroff - specific flux calculation"""

        dt = u.timestep
        cell_sizes = u.cell_sizes
        u1 = u.centroid()

        fluxes = []
        for i in range(3):

            mid_flux = self.flux_function(u, i)

            uplus = u.get_neighbour_state(i, 1)
            uminus = u.get_neighbour_state(i, -1)
            plus_flux = self.flux_function(uplus, i)

            u2 = uplus.centroid()
            intermediate_plus = 0.5 * (
                (u1 + u2) - dt * (plus_flux - mid_flux) / cell_sizes[i]
            )

            minus_flux = self.flux_function(uminus, i)

            u2 = uminus.centroid()
            intermediate_minus = 0.5 * (
                (u1 + u2) - dt * (mid_flux - minus_flux) / cell_sizes[i]
            )

            uplus_star = uplus.copy()
            uplus_star.set_centroid(intermediate_plus)
            uminus_star = uminus.copy()
            uminus_star.set_centroid(intermediate_minus)

            plus_flux = self.flux_function(uplus_star, i)
            minus_flux = self.flux_function(uminus_star, i)
            fluxes.append(plus_flux)
            fluxes.append(minus_flux)
        return fluxes
