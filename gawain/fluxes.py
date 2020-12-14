"""Flux calculation routine

These functions and classes are used for the
physics-specific flux calculations for both 
hydrodynamic and MHD fluxes

"""

import numpy as np


def EulerFluxX(u):
    dens = u.dens()
    momX, momY, momZ = u.momX(), u.momY(), u.momZ()
    en = u.energy()
    pressure = u.pressure()
    x_flux = np.array(
        [
            momX,
            momX * momX / dens + pressure,
            momX * momY / dens,
            momX * momZ / dens,
            (en + pressure) * momX / dens,
        ]
    )
    return x_flux


def EulerFluxY(u):
    dens = u.dens()
    momX, momY, momZ = u.momX(), u.momY(), u.momZ()
    en = u.energy()
    pressure = u.pressure()
    y_flux = np.array(
        [
            momY,
            momY * momX / dens,
            momY * momY / dens + pressure,
            momY * momZ / dens,
            (en + pressure) * momY / dens,
        ]
    )

    return y_flux


def EulerFluxZ(u):
    dens = u.dens()
    momX, momY, momZ = u.momX(), u.momY(), u.momZ()
    en = u.energy()
    pressure = u.pressure()

    z_flux = np.array(
        [
            momZ,
            momZ * momX / dens,
            momZ * momY / dens,
            momZ * momZ / dens + pressure,
            (en + pressure) * momZ / dens,
        ]
    )
    return z_flux


def MHDFluxX(u):
    dens = u.dens()
    momX, momY, momZ = u.momX(), u.momY(), u.momZ()
    en = u.energy()
    tpressure = u.total_pressure()
    bx, by, bz = u.magX(), u.magY(), u.magZ()
    zeros = np.zeros(dens.shape)
    x_flux = np.array(
        [
            momX,
            momX * momX / dens - bx * bx + tpressure,
            momX * momY / dens - bx * by,
            momX * momZ / dens - bx * bz,
            (en + tpressure) * momX / dens
            - (bx * momX + by * momY + bz * momZ) * bx / dens,
            zeros,
            (momX * by - momY * bx) / dens,
            (momX * bz - momZ * bx) / dens,
        ]
    )
    return x_flux


def MHDFluxY(u):
    dens = u.dens()
    momX, momY, momZ = u.momX(), u.momY(), u.momZ()
    en = u.energy()
    tpressure = u.total_pressure()
    bx, by, bz = u.magX(), u.magY(), u.magZ()
    zeros = np.zeros(dens.shape)
    y_flux = np.array(
        [
            momY,
            momY * momX / dens - by * bx,
            momY * momY / dens - by * by + tpressure,
            momY * momZ / dens - by * bz,
            (en + tpressure) * momY / dens
            - (bx * momX + by * momY + bz * momZ) * by / dens,
            (momY * bx - momX * by) / dens,
            zeros,
            (momY * bz - momZ * by) / dens,
        ]
    )
    return y_flux


def MHDFluxZ(u):
    dens = u.dens()
    momX, momY, momZ = u.momX(), u.momY(), u.momZ()
    en = u.energy()
    tpressure = u.total_pressure()
    bx, by, bz = u.magX(), u.magY(), u.magZ()
    zeros = np.zeros(dens.shape)
    z_flux = np.array(
        [
            momZ,
            momZ * momX / dens - bz * bx,
            momZ * momY / dens - bz * by,
            momZ * momZ / dens - bz * bz + tpressure,
            (en + tpressure) * momZ / dens
            - (bx * momX + by * momY + bz * momZ) * bz / dens,
            (momZ * bx - momX * bz) / dens,
            (momZ * by - momY * bz) / dens,
            zeros,
        ]
    )
    return z_flux


class FluxCalculator:
    def __init__(self):
        self.x_plus_flux = None
        self.x_minus_flux = None
        self.y_plus_flux = None
        self.y_minus_flux = None
        self.z_plus_flux = None
        self.z_minus_flux = None
        self.flux_functionX = None
        self.flux_functionY = None
        self.flux_functionZ = None

    def set_flux_function(self, with_mhd):
        if with_mhd:
            self.flux_functionX = MHDFluxX
            self.flux_functionY = MHDFluxY
            self.flux_functionZ = MHDFluxZ
        else:
            self.flux_functionX = EulerFluxX
            self.flux_functionY = EulerFluxY
            self.flux_functionZ = EulerFluxZ

    def _specific_fluxes(self, u):
        pass

    def calculate_flux_divergence(self, u):
        self.x_plus_flux = self.flux_functionX(u.plusX())
        self.x_minus_flux = self.flux_functionX(u.minusX())

        self.y_plus_flux = self.flux_functionY(u.plusY())
        self.y_minus_flux = self.flux_functionY(u.minusY())

        self._specific_fluxes(u)

        total_flux = -(self.y_plus_flux - self.y_minus_flux) / u.dy
        total_flux += -(self.x_plus_flux - self.x_minus_flux) / u.dx
        return total_flux


class HLLFluxer(FluxCalculator):
    """ A MUSCL-Hancock HLL solver using minmod limiter
    """

    def __init__(self):
        super(HLLFluxer, self).__init__()

    def superbee(self, r):
        return np.maximum(
            np.zeros(r.shape),
            np.maximum(
                np.minimum(2 * r, np.ones(r.shape)), np.minimum(r, 2 * np.ones(r.shape))
            ),
        )

    def vanleer(self, r):
        abs_r = np.absolute(r)
        return (r + abs_r) / (1.0 + abs_r)

    def minmod(self, a, b):
        return np.where(
            a * b <= 0.0, 0.0, np.where(np.absolute(a) < np.absolute(b), a, b)
        )

    def wave_speeds_X(self, Ul, Ur):

        lambda_L_min, lambda_L_max = Ul.calculate_min_max_wave_speeds_X()
        lambda_R_min, lambda_R_max = Ur.calculate_min_max_wave_speeds_X()

        zeros = np.zeros(lambda_L_min.shape)

        return (
            np.minimum(np.minimum(zeros, lambda_L_min), lambda_R_min),
            np.maximum(np.maximum(zeros, lambda_L_max), lambda_R_max),
        )

    def wave_speeds_Y(self, Ul, Ur):

        lambda_L_min, lambda_L_max = Ul.calculate_min_max_wave_speeds_Y()
        lambda_R_min, lambda_R_max = Ur.calculate_min_max_wave_speeds_Y()

        zeros = np.zeros(lambda_L_min.shape)

        return (
            np.minimum(np.minimum(zeros, lambda_L_min), lambda_R_min),
            np.maximum(np.maximum(zeros, lambda_L_max), lambda_R_max),
        )

    def hll_flux_X(self, Sl, Sr, Ul, Ur):

        fl = self.flux_functionX(Ul)
        fr = self.flux_functionX(Ur)

        ur = Ur.centroid()
        ul = Ul.centroid()

        fhll = (Sr * fl - Sl * fr + Sl * Sr * (ur - ul)) / (Sr - Sl)

        return np.where(Sl >= 0.0, fl, np.where(Sr <= 0.0, fr, fhll))

    def hll_flux_Y(self, Sl, Sr, Ul, Ur):

        fl = self.flux_functionY(Ul)
        fr = self.flux_functionY(Ur)

        ur = Ur.centroid()
        ul = Ul.centroid()

        fhll = (Sr * fl - Sl * fr + Sl * Sr * (ur - ul)) / (Sr - Sl)

        return np.where(Sl >= 0.0, fl, np.where(Sr <= 0.0, fr, fhll))

    def MUSCL_Hancock_reconstructionX(self, U_left, U_mid, U_right):
        """" returns the states to the rights and lefts of the interfaces
        
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
        self.x_plus_flux = self.flux_functionX(rights)
        self.x_minus_flux = self.flux_functionX(lefts)

        dt = U_mid.timestep
        dx = U_mid.dx

        a = a + 0.5 * dt * (self.x_plus_flux - self.x_minus_flux) / dx
        b = b + 0.5 * dt * (self.x_plus_flux - self.x_minus_flux) / dx

        rights.set_centroid(a)
        lefts.set_centroid(b)

        return lefts, rights

    def MUSCL_Hancock_reconstructionY(self, U_left, U_mid, U_right):
        """" returns the states to the rights and lefts of the interfaces
        
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
        self.y_plus_flux = self.flux_functionY(rights)
        self.y_minus_flux = self.flux_functionY(lefts)

        dt = U_mid.timestep
        dy = U_mid.dy

        a = a + 0.5 * dt * (self.y_plus_flux - self.y_minus_flux) / dy
        b = b + 0.5 * dt * (self.y_plus_flux - self.y_minus_flux) / dy

        rights.set_centroid(a)
        lefts.set_centroid(b)

        return lefts, rights

    def _specific_fluxes(self, u):
        # MUSCL-Hancock reconstruction

        uplus = u.plusX()
        uminus = u.minusX()

        lefts, rights = self.MUSCL_Hancock_reconstructionX(uminus, u, uplus)

        # HLL flux calculation

        # minus flux calculation

        umid = rights
        uminus = lefts.minusX()

        Sl, Sr = self.wave_speeds_X(uminus, umid)

        self.x_minus_flux = self.hll_flux_X(Sl, Sr, uminus, umid)

        # plus flux calculation

        umid = lefts
        uplus = rights.plusX()

        Sl, Sr = self.wave_speeds_X(umid, uplus)

        self.x_plus_flux = self.hll_flux_X(Sl, Sr, umid, uplus)

        #### Y

        uplus = u.plusY()
        uminus = u.minusY()

        lefts, rights = self.MUSCL_Hancock_reconstructionY(uminus, u, uplus)

        # HLL flux calculation

        # minus flux calculation

        umid = rights
        uminus = lefts.minusY()

        Sl, Sr = self.wave_speeds_Y(uminus, umid)

        self.y_minus_flux = self.hll_flux_Y(Sl, Sr, uminus, umid)

        # plus flux calculation

        umid = lefts
        uplus = rights.plusY()

        Sl, Sr = self.wave_speeds_Y(umid, uplus)

        self.y_plus_flux = self.hll_flux_Y(Sl, Sr, umid, uplus)


class LaxFriedrichsFluxer(FluxCalculator):
    def __init__(self):
        super(LaxFriedrichsFluxer, self).__init__()

    def _specific_fluxes(self, u):

        u1 = u.centroid()
        dt = u.timestep
        dx, dy, dz = u.dx, u.dy, u.dz

        mid_flux_x = self.flux_functionX(u)

        u2 = u.minusX().centroid()
        self.x_minus_flux = 0.5 * (
            (self.x_minus_flux + mid_flux_x) - dx * (u1 - u2) / dt
        )

        u2 = u.plusX().centroid()
        self.x_plus_flux = 0.5 * ((self.x_plus_flux + mid_flux_x) - dx * (u2 - u1) / dt)

        mid_flux_y = self.flux_functionY(u)

        u2 = u.minusY().centroid()
        self.y_minus_flux = 0.5 * (
            (self.y_minus_flux + mid_flux_y) - dy * (u1 - u2) / dt
        )

        u2 = u.plusY().centroid()
        self.y_plus_flux = 0.5 * ((self.y_plus_flux + mid_flux_y) - dy * (u2 - u1) / dt)


class LaxWendroffFluxer(FluxCalculator):
    """ using two-step richtmyer method
    """

    def __init__(self):
        super(LaxWendroffFluxer, self).__init__()

    def _specific_fluxes(self, u):

        dt = u.timestep
        dx, dy, dz = u.dx, u.dy, u.dz
        u1 = u.centroid()

        ####

        mid_flux_x = self.flux_functionX(u)

        uplus = u.plusX()
        uminus = u.minusX()

        u2 = uplus.centroid()
        intermediate_plus_x = 0.5 * (
            (u1 + u2) - dt * (self.x_plus_flux - mid_flux_x) / dx
        )

        u2 = uminus.centroid()
        intermediate_minus_x = 0.5 * (
            (u1 + u2) - dt * (mid_flux_x - self.x_minus_flux) / dx
        )

        uplus_star = uplus.copy()
        uplus_star.set_centroid(intermediate_plus_x)
        uminus_star = uminus.copy()
        uminus_star.set_centroid(intermediate_minus_x)

        self.x_plus_flux = self.flux_functionX(uplus_star)
        self.x_minus_flux = self.flux_functionX(uminus_star)

        ####

        mid_flux_y = self.flux_functionY(u)

        uplus = u.plusY()
        uminus = u.minusY()

        u2 = uplus.centroid()
        intermediate_plus_y = 0.5 * (
            (u1 + u2) - dt * (self.y_plus_flux - mid_flux_y) / dy
        )

        u2 = uminus.centroid()
        intermediate_minus_y = 0.5 * (
            (u1 + u2) - dt * (mid_flux_y - self.y_minus_flux) / dy
        )

        uplus_star = uplus.copy()
        uplus_star.set_centroid(intermediate_plus_y)
        uminus_star = uminus.copy()
        uminus_star.set_centroid(intermediate_minus_y)

        self.y_plus_flux = self.flux_functionY(uplus_star)
        self.y_minus_flux = self.flux_functionY(uminus_star)
