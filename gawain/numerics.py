''' Numerical utilities '''

import time

from tqdm import tqdm

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
        self.integrator = Integrator(self.data, Parameters.cell_sizes)
        
    def update(self, delta):
        self.data = self.integrator.integrate(self.data, delta)



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
        
    def predictor_corrector(self, solution_data, h):
        intermediate_flux = self.fluxer.calculate_fluxes(solution_data, h)
        intermediate_solution = solution_data + h*intermediate_flux
        final_flux = self.fluxer.calculate_fluxes(intermediate_solution, h)
        solution_data += 0.5*h*(intermediate_flux + final_flux)

class LeapFrogIntegrator(Integrator):
    def __init__(self):
        super(LeapFrogIntegrator, self).__init__()
        
    def leapfrog(self, solution_data, h):
        dummy = solution_data
        intermediate_flux = self.fluxer.calculate_fluxes(solution_data, h)
        solution_data = self.lagged_solution+h*intermediate_flux
        self.lagged_solution = dummy
        return 

class RK2Integrator(Integrator):
    def __init__(self):
        super(RK2Integrator, self).__init__()
        
    def RK2(self, solution_data, h):
        k1 = self.fluxer.calculate_fluxes(solution_data, h)
        k1 *= h
        k2 = self.fluxer.calculate_fluxes(solution_data + 0.75*k1, h)
        k2 *= h
        solution_data += 0.333*k1 + 0.666*k2

        return solution_data
        
class BoundaryConditions:
    def __init__(self):
        pass