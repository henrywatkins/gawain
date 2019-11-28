''' Numerical utilities '''



class Clock:
    def.__init__(self, Parameters):
        self.current_time = 0.0
        self.end_time = Parameters.period
        self.timestep = 0.001
        self.next_output_time = 0.0
        self.output_spacing = self.end_time/Parameters.n_outputs

    def is_end():
        if self.current_time<self.end_time:
            return False
        else:
            return True

    def update():
        self.current_time += self.timestep

    def is_output():
        if self.current_time >= self.next_output_time:
            self.next_output_time += self.output_spacing
            return True
        else:
            return False


    def calculate_timestep(SolutionVector):

        self.timestep = dt
        return dt


class SolutionVector:
    def.__init__(self, Parameters):
        self.data = Parameters.initial_condition
        self.

    def update():
        pass

    def calculate_fluxes():
        pass
