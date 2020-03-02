''' physics-specific routines '''

import numpy as np

def BurgersFlux(array):
    speed=0.1
    return speed*array*array

def EulerFluxX(SolutionVector):

    pressure = SolutionVector.adi_minus1()*(SolutionVector.en()-0.5*SolutionVector.momMagSqr()/SolutionVector.dens())

    x_flux = np.array(SolutionVector.momX(),
                      SolutionVector.momX()*SolutionVector.momX()/SolutionVector.dens()+pressure,
                      SolutionVector.momX()*SolutionVector.momY()/SolutionVector.dens(),
                      (SolutionVector.en() + pressure)*SolutionVector.momX()/SolutionVector.dens())
    return x_fluz

def EulerFluxY(SolutionVector):

    pressure = SolutionVector.adi_minus1()*(SolutionVector.en()-0.5*SolutionVector.momMagSqr()/SolutionVector.dens())

    y_flux = np.array(SolutionVector.momY(),
                      SolutionVector.momY()*SolutionVector.momY()/SolutionVector.dens()+pressure,
                      SolutionVector.momX()*SolutionVector.momY()/SolutionVector.dens(),
                      (SolutionVector.en() + pressure)*SolutionVector.momY()/SolutionVector.dens())

    return y_flux

def EulerFluxZ(SolutionVector):
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

    def specific_fluxes(self, SolutionVector):
        pass

    def calculate_rhs(self, SolutionVector):
        self.x_plus_flux = self.flux_functionX(SolutionVector.plusX())
        self.x_minus_flux = self.flux_functionX(SolutionVector.minusX())

        self.y_plus_flux = self.flux_functionY(SolutionVector.plusY())
        self.y_minus_flux = self.flux_functionY(SolutionVector.minusY())

        self.specific_fluxes(SolutionVector)

        total_flux = -(self.y_plus_flux - self.y_minus_flux)/self.dy
        total_flux += -(self.x_plus_flux - self.x_minus_flux)/self.dx
        return total_flux



class HLLFluxer(FluxCalculator):
    def __init__(self):
        pass

class LaxFriedrichsFluxer(FluxCalculator):
    def __init__(self, Parameters):
        super(LaxFriedrichsFluxer, self).__init__(Parameters)
    def specific_fluxes(self, SolutionVector):

       u1 = SolutionVector.centroid()

       mid_flux_x = self.flux_functionX(u1)
       mid_flux_y = self.flux_functionY(u1)

       self.x_minus_flux = 0.5*(self.x_minus_flux+mid_flux_x)
       u2 = SolutionVector.minusX()
       self.x_minus_flux+=-0.5*self.dx*1000.*(u1-u2)

       self.x_plus_flux = 0.5*(self.x_plus_flux+mid_flux_x)
       u2 = SolutionVector.plusX()
       self.x_plus_flux+=-0.5*self.dx*1000.*(u2-u1)

       self.y_minus_flux = 0.5*(self.y_minus_flux+mid_flux_y)
       u2 = SolutionVector.minusY()
       self.y_minus_flux+=-0.5*self.dx*1000.*(u1-u2)

       self.y_plus_flux = 0.5*(self.y_plus_flux+mid_flux_y)
       u2 = SolutionVector.plusY()
       self.y_plus_flux+=-0.5*self.dx*1000.*(u2-u1)


class LaxWendroffFluxer(FluxCalculator):
    """ using two-step richtmyer method
    """
    def __init__(self, Parameters):
        super(LaxWendroffFluxer, self).__init__(Parameters)
    def specific_fluxes(self, SolutionVector):

       u1 = SolutionVector.centroid()

       mid_flux_x = self.flux_functionX(u1)
       mid_flux_y = self.flux_functionY(u1)

       u2 = SolutionVector.minusX()
       intermediate_plus_x = 0.5*(u1+u2)
       intermediate_plus_x+=-0.5*self.dx*1000.*(self.x_plus_flux-mid_flux_x)

       u2 = SolutionVector.plusX()
       intermediate_minus_x = 0.5*(u1+u2)
       intermediate_minus_x+=-0.5*self.dx*1000.*(mid_flux_x-self.x_minus_flux)

       self.x_plus_flux = self.flux_functionX(intermediate_plus_x)
       self.x_minus_flux = self.flux_functionX(intermediate_minus_x)

       u2 = SolutionVector.minusY()
       intermediate_plus_y = 0.5*(u1+u2)
       intermediate_plus_y+=-0.5*self.dx*1000.*(self.y_plus_flux-mid_flux_y)

       u2 = SolutionVector.plusY()
       intermediate_minus_y = 0.5*(u1+u2)
       intermediate_minus_y+=-0.5*self.dx*1000.*(mid_flux_y-self.y_minus_flux)

       self.y_plus_flux = self.flux_functionY(intermediate_plus_y)
       self.y_minus_flux = self.flux_functionY(intermediate_minus_y)


class MUSCLFluxer(FluxCalculator):

    def MUSCL(self, solution_data, dt):
        pass
