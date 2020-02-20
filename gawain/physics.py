''' physics-specific routines '''

import numpy as np

def F(array):
    speed = 0.1
    return speed*array


class FluxCalculator:
    def __init__(self, Parameters):
        self.cell_sizes = Parameters.cell_sizes
        self.flux_function = F

    def calculate_fluxes(self, SolutionVector):
        """ Returns the RHS of the conservation equation

        make so that it is agnostic to the exact form of flux function


        """
        x_plus_flux = self.flux_function(SolutionVector.plusX())
        x_minus_flux = self.flux_function(SolutionVector.minusX())
        total_flux = -(x_plus_flux - x_minus_flux)/self.cell_sizes[0]
        return total_flux


class HLLFluxer(FluxCalculator):
    def __init__(self):
        pass

class LaxFriedrichsFluxer(FluxCalculator):

   def lax_friedrichs(self, solution_data, dt):

        # using periodic conditions for now
        shifted_left = np.roll(solution_data, -1, axis=0)
        shifted_right = np.roll(solution_data, 1, axis=0)
        shifted_up = np.roll(solution_data, 1, axis=1)
        shifted_down = np.roll(solution_data, -1, axis=1)

        middle_x = self.x_flux(solution_data)
        middle_y = self.y_flux(solution_data)
        left = self.x_flux(shifted_right)
        right = self.x_flux(shifted_left)
        top = self.y_flux(shifted_down)
        bottom = self.y_flux(shifted_up)

        x_minus_flux = 0.5*(left)#-0.5*self.cell_sizes[0]*(solution_data - shifted_right)/dt
        x_plus_flux = 0.5*(right)#-0.5*self.cell_sizes[0]*(solution_data - shifted_left)/dt

        y_minus_flux = 0.5*(bottom)#-0.5*self.cell_sizes[1]*(solution_data - shifted_down)/dt
        y_plus_flux = 0.5*(top)#-0.5*self.cell_sizes[1]*(solution_data - shifted_up)/dt

        return x_plus_flux, x_minus_flux, y_plus_flux, y_minus_flux


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
