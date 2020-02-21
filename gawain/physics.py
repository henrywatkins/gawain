''' physics-specific routines '''

import numpy as np

def F(array):
    speed = 0.1
    return speed*array*array


class FluxCalculator:
    def __init__(self, Parameters):
        (self.dx, self.dy, self.dz) = Parameters.cell_sizes
        self.flux_function = F

    def calculate_fluxes(self, SolutionVector):
        """ Returns the RHS of the conservation equation

        make so that it is agnostic to the exact form of flux function


        """
        x_plus_flux = self.flux_function(SolutionVector.plusX())
        x_minus_flux = self.flux_function(SolutionVector.minusX())
        total_flux = -(x_plus_flux - x_minus_flux)/self.dx
        return total_flux


class HLLFluxer(FluxCalculator):
    def __init__(self):
        pass

class LaxFriedrichsFluxer(FluxCalculator):
    def __init__(self, Parameters):
        super(LaxFriedrichsFluxer, self).__init__(Parameters)
    def calculate_fluxes(self, SolutionVector):
       x_plus_flux = self.flux_function(SolutionVector.plusX())
       x_mid_flux = self.flux_function(SolutionVector.centroid())
       x_minus_flux = self.flux_function(SolutionVector.minusX())

       x_minus_flux = 0.5*(x_minus_flux+x_mid_flux)
       ui, ui1 = SolutionVector.centroid(), SolutionVector.minusX()
       x_minus_flux+=-0.5*self.dx*1000.*(ui-ui1)

       x_plus_flux = 0.5*(x_plus_flux+x_mid_flux)
       ui, ui1 = SolutionVector.centroid(), SolutionVector.plusX()
       x_plus_flux+=-0.5*self.dx*1000.*(ui1-ui)

       total_flux = -(x_plus_flux - x_minus_flux)/self.dx
       return total_flux

class LaxWendroffFluxer(FluxCalculator):
    """ using two-step richtmyer method
    """
    def __init__(self, Parameters):
        super(LaxWendroffFluxer, self).__init__(Parameters)
    def calculate_fluxes(self, SolutionVector):
       x_plus_flux = self.flux_function(SolutionVector.plusX())
       x_mid_flux = self.flux_function(SolutionVector.centroid())
       x_minus_flux = self.flux_function(SolutionVector.minusX())

       ui, ui1 = SolutionVector.centroid(), SolutionVector.minusX()
       intermediate_plus = 0.5*(ui+ui1)
       intermediate_plus+=-0.5*self.dx*1000.*(x_plus_flux-x_mid_flux)

       ui, ui1 = SolutionVector.centroid(), SolutionVector.plusX()
       intermediate_minus = 0.5*(ui+ui1)
       intermediate_minus+=-0.5*self.dx*1000.*(x_mid_flux-x_minus_flux)

       x_plus_flux = self.flux_function(intermediate_plus)
       x_minus_flux = self.flux_function(intermediate_minus)
       total_flux = -(x_plus_flux - x_minus_flux)/self.dx
       return total_flux

class MUSCLFluxer(FluxCalculator):

    def MUSCL(self, solution_data, dt):

        # using periodic conditions for now
        shifted_left = np.roll(solution_data, -1, axis=0)
        shifted_right = np.roll(solution_data, 1, axis=0)
        shifted_up = np.roll(solution_data, 1, axis=1)
        shifted_down = np.roll(solution_data, -1, axis=1)

        reconstruct_x_right = 0.5*(solution_data + shifted_left)
        reconstruct_x_left = 0.5*(solution_data + shifted_right)
        reconstruct_y_top = 0.5*(solution_data + shifted_down)
        reconstruct_y_bottom = 0.5*(solution_data + shifted_up)

        x_minus_flux = self.x_flux(reconstruct_x_left)
        x_plus_flux = self.x_flux(reconstruct_x_right)

        y_minus_flux = self.y_flux(reconstruct_y_bottom)
        y_plus_flux = self.y_flux(reconstruct_y_top)

        return x_plus_flux, x_minus_flux, y_plus_flux, y_minus_flux
