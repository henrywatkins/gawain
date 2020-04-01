''' physics-specific routines '''

import numpy as np

def BurgersFlux(array):
    speed=0.1
    return speed*array*array

def EulerFluxX(u):

    """
    thermal_en = (u.en()-0.5*u.momMagSqr()/u.dens())
    pressure = u.adi_minus1()*thermal_en

    x_flux = np.array([u.momX(),
                      u.momX()*u.momX()/u.dens()+pressure,
                      u.momX()*u.momY()/u.dens(),
                      u.momX()*u.momZ()/u.dens(),
                      (u.en() + pressure)*u.momX()/u.dens()])
    """
    dens = u[0]
    momX = u[1]
    momY = u[2]
    momZ = u[3]
    en = u[4]
    adi_minus1 = 0.4
    momMagSqr = momX*momX + momY*momY + momZ*momZ

    thermal_en = (en-0.5*momMagSqr/dens)
    pressure = adi_minus1*thermal_en

    x_flux = np.array([momX,
                      momX*momX/dens+pressure,
                      momX*momY/dens,
                      momX*momZ/dens,
                      (en + pressure)*momX/dens])
    return x_flux

def EulerFluxY(u):
    """
    thermal_en = (u.en()-0.5*u.momMagSqr()/u.dens())
    pressure = u.adi_minus1()*thermal_en

    y_flux = np.array([u.momY(),
                      u.momY()*u.momX()/u.dens(),
                      u.momY()*u.momY()/u.dens()+pressure,
                      u.momY()*u.momZ()/u.dens(),
                      (u.en() + pressure)*u.momY()/u.dens()])
    """
    dens = u[0]
    momX = u[1]
    momY = u[2]
    momZ = u[3]
    en = u[4]
    adi_minus1 = 0.4
    momMagSqr = momX*momX + momY*momY + momZ*momZ

    thermal_en = (en-0.5*momMagSqr/dens)
    pressure = adi_minus1*thermal_en

    y_flux = np.array([momX,
                      momY*momX/dens,
                      momY*momY/dens+pressure,
                      momY*momZ/dens,
                      (en + pressure)*momY/dens])

    return y_flux

def EulerFluxZ(u):
    pass

class FluxCalculator:
    def __init__(self, Parameters):
        (self.dx, self.dy, self.dz) = Parameters.cell_sizes
        self.x_plus_flux = None
        self.x_minus_flux = None
        self.y_plus_flux = None
        self.y_minus_flux = None
        self.flux_functionX = EulerFluxX
        self.flux_functionY = EulerFluxY

    def _specific_fluxes(self, u):
        pass

    def calculate_rhs(self, u):
        self.x_plus_flux = self.flux_functionX(u.plusX())
        self.x_minus_flux = self.flux_functionX(u.minusX())

        self.y_plus_flux = self.flux_functionY(u.plusY())
        self.y_minus_flux = self.flux_functionY(u.minusY())

        self._specific_fluxes(u)

        total_flux = -(self.y_plus_flux - self.y_minus_flux)/self.dy
        total_flux += -(self.x_plus_flux - self.x_minus_flux)/self.dx
        return total_flux



class HLLFluxer(FluxCalculator):
    def __init__(self):
        pass

class LaxFriedrichsFluxer(FluxCalculator):
    def __init__(self, Parameters):
        super(LaxFriedrichsFluxer, self).__init__(Parameters)
    def _specific_fluxes(self, u):

       u1 = u.centroid()

       mid_flux_x = self.flux_functionX(u1)
       mid_flux_y = self.flux_functionY(u1)

       self.x_minus_flux = 0.5*(self.x_minus_flux+mid_flux_x)
       u2 = u.minusX()
       self.x_minus_flux+=-0.5*self.dx*10000.*(u1-u2)

       self.x_plus_flux = 0.5*(self.x_plus_flux+mid_flux_x)
       u2 = u.plusX()
       self.x_plus_flux+=-0.5*self.dx*10000.*(u2-u1)

       self.y_minus_flux = 0.5*(self.y_minus_flux+mid_flux_y)
       u2 = u.minusY()
       self.y_minus_flux+=-0.5*self.dy*10000.*(u1-u2)

       self.y_plus_flux = 0.5*(self.y_plus_flux+mid_flux_y)
       u2 = u.plusY()
       self.y_plus_flux+=-0.5*self.dy*10000.*(u2-u1)


class LaxWendroffFluxer(FluxCalculator):
    """ using two-step richtmyer method
    """
    def __init__(self, Parameters):
        super(LaxWendroffFluxer, self).__init__(Parameters)
    def _specific_fluxes(self, u):

       u1 = u.centroid()

       mid_flux_x = self.flux_functionX(u1)
       mid_flux_y = self.flux_functionY(u1)

       u2 = u.minusX()
       intermediate_plus_x = 0.5*(u1+u2)
       intermediate_plus_x+=-0.5*self.dx*10000.*(self.x_plus_flux-mid_flux_x)

       u2 = u.plusX()
       intermediate_minus_x = 0.5*(u1+u2)
       intermediate_minus_x+=-0.5*self.dx*10000.*(mid_flux_x-self.x_minus_flux)

       self.x_plus_flux = self.flux_functionX(intermediate_plus_x)
       self.x_minus_flux = self.flux_functionX(intermediate_minus_x)

       u2 = u.minusY()
       intermediate_plus_y = 0.5*(u1+u2)
       intermediate_plus_y+=-0.5*self.dy*10000.*(self.y_plus_flux-mid_flux_y)

       u2 = u.plusY()
       intermediate_minus_y = 0.5*(u1+u2)
       intermediate_minus_y+=-0.5*self.dy*10000.*(mid_flux_y-self.y_minus_flux)

       self.y_plus_flux = self.flux_functionY(intermediate_plus_y)
       self.y_minus_flux = self.flux_functionY(intermediate_minus_y)


class MUSCLFluxer(FluxCalculator):

    def MUSCL(self, solution_data, dt):
        pass
