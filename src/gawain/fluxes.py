"""Flux calculation routine

These functions and classes are used for the
physics-specific flux calculations for both
hydrodynamic and MHD fluxes

"""

from typing import List, Tuple, Union

import numpy as np

from .numerics import MHDSolutionVector, SolutionVector


def EulerFlux(u: Union[SolutionVector, MHDSolutionVector], axis: int) -> np.ndarray:
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


def MHDFlux(u: MHDSolutionVector, axis: int) -> np.ndarray:
    """Calculate the MHD flux along a given axis from a solution vector"""
    dens = u.dens()
    momX, momY, momZ = u.mom(0), u.mom(1), u.mom(2)
    axis_mom = u.mom(axis)
    axis_b = u.mag(axis)
    en = u.energy()
    tpressure = u.total_pressure()
    bx, by, bz = u.mag(0), u.mag(1), u.mag(2)
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
    flux[axis + 5] = 0.0
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

    def __init__(self) -> None:
        self.flux_function = None

    def set_flux_function(self, with_mhd: bool) -> None:
        """set the flux calculation function to euler or mhd"""
        self.flux_function = MHDFlux if with_mhd else EulerFlux

    def _specific_fluxes(
        self, u: Union[SolutionVector, MHDSolutionVector]
    ) -> List[np.ndarray]:
        return [
            self.flux_function(u.get_neighbour_state(i, j), i)
            for i in range(3)
            for j in (1, -1)
        ]

    def calculate_flux_divergence(
        self, u: Union[SolutionVector, MHDSolutionVector]
    ) -> np.ndarray:
        """Flux divergence calculation"""

        fluxes = self._specific_fluxes(u)
        total_flux = -(fluxes[0] - fluxes[1]) / u.dx
        total_flux += -(fluxes[2] - fluxes[3]) / u.dy
        total_flux += -(fluxes[4] - fluxes[5]) / u.dz
        return total_flux


class HLLFluxer(FluxCalculator):
    """A MUSCL-Hancock HLL solver using minmod limiter

    This class calculates the numerical divergence of the flux
    in the conservation equation problem using the HLL solver
    with MUSCL-hancock reconstruction and a minmod limiter.
    """

    def __init__(self) -> None:
        super(HLLFluxer, self).__init__()

    def minmod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """minmod limiter"""
        return np.where(
            a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0 * a
        )

    def wave_speeds(
        self,
        Ul: Union[SolutionVector, MHDSolutionVector],
        Ur: Union[SolutionVector, MHDSolutionVector],
        axis: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the min/max wave speeds at the inteface between two cells"""
        ulminusal, ulplusal = Ul.eigen_speeds(axis)
        urminusal, urplusal = Ur.eigen_speeds(axis)
        return np.minimum(ulminusal, urminusal), np.maximum(ulplusal, urplusal)

    def hll_flux(
        self,
        Sl: np.ndarray,
        Sr: np.ndarray,
        Ul: Union[SolutionVector, MHDSolutionVector],
        Ur: Union[SolutionVector, MHDSolutionVector],
        axis: int,
    ) -> np.ndarray:
        """Calculate the HLLE flux at the interface"""

        fl = self.flux_function(Ul, axis)
        fr = self.flux_function(Ur, axis)

        ur = Ur.centroid()
        ul = Ul.centroid()

        fhll = (Sr * fl - Sl * fr + Sl * Sr * (ur - ul)) / (Sr - Sl)

        return np.select(
            [Sl >= 0.0, Sr <= 0.0, (Sl <= 0.0) & (Sr >= 0.0)], [fl, fr, fhll]
        )

    def MUSCL_Hancock_reconstruction(
        self,
        uminus: Union[SolutionVector, MHDSolutionVector],
        u: Union[SolutionVector, MHDSolutionVector],
        uplus: Union[SolutionVector, MHDSolutionVector],
        axis: int,
    ) -> Tuple[
        Tuple[
            Union[SolutionVector, MHDSolutionVector],
            Union[SolutionVector, MHDSolutionVector],
        ],
        Tuple[
            Union[SolutionVector, MHDSolutionVector],
            Union[SolutionVector, MHDSolutionVector],
        ],
    ]:
        """Calculate the MUSCL-hancock reconstruction of cell values at an cell interface"""

        dt = u.timestep
        delta = u.cell_sizes[axis]

        u_ip1 = uplus.centroid()
        u_i = u.centroid()
        u_im1 = uminus.centroid()
        u_im2 = uminus.get_neighbour_state(axis, -1).centroid()
        u_ip2 = uplus.get_neighbour_state(axis, 1).centroid()

        del_i_right = u_ip1 - u_i
        del_i_left = u_i - u_im1
        sigma_i = self.minmod(del_i_left, del_i_right)

        del_ip1_right = u_ip2 - u_ip1
        del_ip1_left = u_ip1 - u_i
        sigma_ip1 = self.minmod(del_ip1_right, del_ip1_left)

        del_im1_right = u_i - u_im1
        del_im1_left = u_im1 - u_im2
        sigma_im1 = self.minmod(del_im1_left, del_im1_right)

        ul_im1 = u_im1 - 0.5 * sigma_im1
        ur_im1 = u_im1 + 0.5 * sigma_im1

        ul_ip1 = u_ip1 - 0.5 * sigma_ip1
        ur_ip1 = u_ip1 + 0.5 * sigma_ip1

        ul_i = u_i - 0.5 * sigma_i
        ur_i = u_i + 0.5 * sigma_i

        FL_i = self.flux_function(u.copy_with_data(ul_i), axis)
        FR_i = self.flux_function(u.copy_with_data(ur_i), axis)
        FL_ip1 = self.flux_function(u.copy_with_data(ul_ip1), axis)
        FR_ip1 = self.flux_function(u.copy_with_data(ur_ip1), axis)
        FL_im1 = self.flux_function(u.copy_with_data(ul_im1), axis)
        FR_im1 = self.flux_function(u.copy_with_data(ur_im1), axis)

        ul_star_ip1 = ul_ip1 + 0.5 * (dt / delta) * (FL_ip1 - FR_ip1)
        ur_star_i = ur_i + 0.5 * (dt / delta) * (FL_i - FR_i)
        ul_star_i = ul_i + 0.5 * (dt / delta) * (FL_i - FR_i)
        ur_star_im1 = ur_im1 + 0.5 * (dt / delta) * (FL_im1 - FR_im1)

        Ul_right_interface = u.copy_with_data(ur_star_i)
        Ur_right_interface = u.copy_with_data(ul_star_ip1)

        Ul_left_interface = u.copy_with_data(ur_star_im1)
        Ur_left_interface = u.copy_with_data(ul_star_i)

        return (Ul_left_interface, Ur_left_interface), (
            Ul_right_interface,
            Ur_right_interface,
        )

    def hll_directional_fluxes(
        self, u: Union[SolutionVector, MHDSolutionVector], axis: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        uplus1 = u.get_neighbour_state(axis, 1)
        uminus1 = u.get_neighbour_state(axis, -1)
        (ULl, ULr), (URl, URr) = self.MUSCL_Hancock_reconstruction(
            uminus1, u, uplus1, axis
        )

        Sl, Sr = self.wave_speeds(URl, URr, axis)
        right_interface_flux = self.hll_flux(Sl, Sr, URl, URr, axis)

        Sl, Sr = self.wave_speeds(ULl, ULr, axis)
        left_interface_flux = self.hll_flux(Sl, Sr, ULl, ULr, axis)

        return right_interface_flux, left_interface_flux

    def _specific_fluxes(
        self, u: Union[SolutionVector, MHDSolutionVector]
    ) -> List[np.ndarray]:
        """HLL solver - specifc flux calculation"""
        fluxes = []
        for i in range(3):
            fluxes.extend(self.hll_directional_fluxes(u, i))
        return fluxes


class LaxFriedrichsFluxer(FluxCalculator):
    """Lax-Friedrichs method flux calculator

    This class calculates the numerical divergence of the flux
    in the conservation equation problem using the Lax-Friedrichs method.
    """

    def __init__(self) -> None:
        super(LaxFriedrichsFluxer, self).__init__()

    def _specific_fluxes(
        self, u: Union[SolutionVector, MHDSolutionVector]
    ) -> List[np.ndarray]:
        """Lax-Friedrichs -specific flux calculation"""
        u1 = u.centroid()
        dt = u.timestep
        cell_sizes = u.cell_sizes

        fluxes = []
        for i in range(3):
            mid_flux = self.flux_function(u, i)
            u2 = u.get_neighbour_state(i, 1).centroid()
            plus_flux = self.flux_function(u.get_neighbour_state(i, 1), i)
            plus_flux = 0.5 * ((plus_flux + mid_flux) - cell_sizes[i] * (u2 - u1) / dt)
            fluxes.append(plus_flux)
            u2 = u.get_neighbour_state(i, -1).centroid()
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

    def __init__(self) -> None:
        super(LaxWendroffFluxer, self).__init__()

    def _specific_fluxes(
        self, u: Union[SolutionVector, MHDSolutionVector]
    ) -> List[np.ndarray]:
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

            uplus_star = uplus.copy_with_data(intermediate_plus)
            uminus_star = uminus.copy_with_data(intermediate_minus)

            plus_flux = self.flux_function(uplus_star, i)
            minus_flux = self.flux_function(uminus_star, i)
            fluxes.append(plus_flux)
            fluxes.append(minus_flux)
        return fluxes
