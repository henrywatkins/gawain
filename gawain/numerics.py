''' Numerical utilities '''

import time

from progressbar import ProgressBar

from gawain.physics import calculate_fluxes

class Clock:
    def __init__(self, Parameters):
        self.current_time = 0.0
        self.end_time = Parameters.t_max
        self.timestep = 0.001
        self.next_output_time = 0.0
        self.output_spacing = self.end_time/Parameters.n_outputs
        self.bar = ProgressBar(max_value=self.end_time)
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
    def duration():
        wallclock_end = time.process_time()
        return wallclock_end - self.wallclock_start


    def calculate_timestep(self, SolutionVector):

        dt = 0.000001

        self.timestep = dt
        return dt


class SolutionVector:
    def __init__(self, Parameters):
        self.data = Parameters.initial_condition
        self.integrate = Integrator(self.data)

    def update(self, delta):
        #integrate.leapfrog(self.data, delta)
        pass


class Integrator:
    def __init__(self, solution_data):
        self.lagged_solution = SolutionVector.data


    def leapfrog(self, solution_data, h):
        dummy = solution_data
        calculate_fluxes(solution_data)
        solution_data = self.lagged_solution+h*solution_data
        self.lagged_solution = dummy

    def RK2(self, solution_data, h):
        k1 = solution_data
        calculate_fluxes(k1)
        k1 = h*k1
        k2 = solution_data + 0.75*k1
        calculate_fluxes(k2)
        k2 = h*k2
        solution_data += 0.333*k1 + 0.666*k2

    def RKL2(self, solution_data, h):
        pass

class BoundaryConditions:
    def __init__(self):
        pass
