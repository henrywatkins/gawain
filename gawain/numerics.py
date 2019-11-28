''' Numerical utilities '''

from progressbar import ProgressBar

class Clock:
    def __init__(self, Parameters):
        self.current_time = 0.0
        self.end_time = Parameters.t_max
        self.timestep = 0.001
        self.next_output_time = 0.0
        self.output_spacing = self.end_time/Parameters.n_outputs
        self.bar = ProgressBar(max_value=self.end_time)

    def is_end(self):
        if self.current_time<self.end_time:
            return False
        else:
            return True

    def update(self):
        self.bar.update(self.current_time)
        self.current_time += self.timestep

    def is_output(self):
        if self.current_time >= self.next_output_time:
            self.next_output_time += self.output_spacing
            return True
        else:
            return False


    def calculate_timestep(self, SolutionVector):

        dt = 0.000001

        self.timestep = dt
        return dt


class SolutionVector:
    def __init__(self, Parameters):
        self.data = Parameters.initial_condition
        self.thing = 0

    def update(self, delta):
        pass

    def calculate_fluxes(self):
        pass
