''' Numerical utilities '''

import time

from tqdm import tqdm
import numpy as np

from gawain.physics import FluxCalculator

class Clock:
    def __init__(self, Parameters):
        self.current_time = 0.0
        self.end_time = Parameters.t_max
        self.cfl = Parameters.cfl
        self.timestep = 0.001
        self.next_output_time = 0.0
        self.output_spacing = self.end_time/Parameters.n_outputs
        self.bar = tqdm(total=self.end_time)
        self.wallclock_start = time.process_time()

    def is_end(self):
        if self.current_time<self.end_time:
            return False
        else:
            return True

    def tick(self):
        self.bar.update(self.current_time)
        self.current_time += self.timestep

    def is_output(self):
        if self.current_time >= self.next_output_time:
            self.next_output_time += self.output_spacing
            return True
        else:
            return False


    def duration(self):
        wallclock_end = time.process_time()
        return wallclock_end - self.wallclock_start


    def calculate_timestep(self, SolutionVector):

        dt = 0.01
        self.timestep = dt
        return dt


class SolutionVector:
    def __init__(self, Parameters):
        self.data = Parameters.initial_condition
        #self.integrator = PredictorCorrectorIntegrator(self.data, Parameters.cell_sizes)
        #self.integrator = Integrator(self.data, Parameters.cell_sizes)
        self.integrator = LeapFrogIntegrator(self.data, Parameters.cell_sizes)
        #self.integrator = RK2Integrator(self.data, Parameters.cell_sizes)
        self.boundary_condition = Parameters.boundary_conditions
    
    def set_centroid(self, array):
        self.data = array
        
    def centroid(self):
        return self.data
    
    def plusX(self):
        """ returns data shifted i+1
        """
        if self.boundary_condition[0]=="periodic":
            return np.roll(self.data, 1)
    
    def minusX(self):
        """ returns data shifted i-1
        """
        if self.boundary_condition[0]=="periodic":
            return np.roll(self.data, -1)
        
    def plusY(self):
        """ returns data shifted j+1
        """
        pass
        
    def minusY(self):
        """ returns data shifted j-1
        """
        pass
 
    def update(self, time_step):
        self.data = self.integrator.integrate(self.data, time_step)

class Integrator:
    def __init__(self, solution_data, cell_sizes):
        self.lagged_solution = solution_data
        self.fluxer = FluxCalculator(cell_sizes)

    def integrate(self, solution_data, time_step):
        rhs = self.fluxer.calculate_fluxes(solution_data)
        return solution_data + time_step*rhs


class PredictorCorrectorIntegrator(Integrator):
    def __init__(self, solution_data, cell_sizes):
        super(PredictorCorrectorIntegrator, self).__init__(solution_data, cell_sizes)

    def integrate(self, solution_data, time_step):
        intermediate_flux = self.fluxer.calculate_fluxes(solution_data)
        intermediate_solution = solution_data + time_step*intermediate_flux
        final_flux = self.fluxer.calculate_fluxes(intermediate_solution)
        return solution_data + 0.5*time_step*(intermediate_flux + final_flux)

class LeapFrogIntegrator(Integrator):
    def __init__(self, solution_data, cell_sizes):
        super(LeapFrogIntegrator, self).__init__(solution_data, cell_sizes)

    def integrate(self, solution_data, time_step):
        dummy = solution_data
        intermediate_flux = self.fluxer.calculate_fluxes(solution_data)
        solution_data = self.lagged_solution+2.*time_step*intermediate_flux
        self.lagged_solution = dummy
        return solution_data

class RK2Integrator(Integrator):
    def __init__(self, solution_data, cell_sizes):
        super(RK2Integrator, self).__init__(solution_data, cell_sizes)

    def integrate(self, solution_data, time_step):
        k1 = self.fluxer.calculate_fluxes(solution_data)
        k1 *= time_step
        k2 = self.fluxer.calculate_fluxes(solution_data + 0.75*k1)
        k2 *= time_step
        solution_data += 0.333*k1 + 0.666*k2
        return solution_data

class BoundaryConditions:
    def __init__(self, params):
        self.type = params.boundary_conditions
        pass
