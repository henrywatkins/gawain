''' physics-specific routines '''

import numpy as np

class FluxCalculator:
    def __init__(self, cell_sizes):
        self.cell_sizes = cell_sizes

    def calculate_fluxes(self, solution_data, dt):

        x_plus_flux, x_minus_flux, y_plus_flux, y_minus_flux = self.MUSCL(solution_data, dt)
        total_flux = (x_plus_flux - x_minus_flux)/self.cell_sizes[0]
        #total_flux += (y_plus_flux - y_minus_flux)/self.cell_sizes[1]
        return total_flux

    def x_flux(self, vector):
        return 0.1*vector

    def y_flux(self, vector):
        return 0.0*vector
        
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
