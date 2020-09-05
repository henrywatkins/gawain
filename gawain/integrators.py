""" integrator classes
"""

import numpy as np


class Integrator:
    def __init__(self, solution_vector, parameters):
        self.lagged_solution = solution_vector
        self.fluxer = parameters.fluxer_type()

    def integrate(self, solution_vector):
        intermediate_rhs = self.fluxer.calculate_rhs(solution_vector)
        solution_vector.update(intermediate_rhs)
        return solution_vector


class PredictorCorrectorIntegrator(Integrator):
    def __init__(self, solution_vector, parameters):
        super(PredictorCorrectorIntegrator, self).__init__(solution_vector, parameters)

    def integrate(self, solution_vector):
        intermediate_rhs = self.fluxer.calculate_rhs(solution_vector)
        intermediate_solution = solution_vector
        intermediate_solution.update(intermediate_rhs)
        final_rhs = self.fluxer.calculate_rhs(intermediate_solution)
        final_rhs = 0.5 * (intermediate_rhs + final_rhs)
        solution_vector.update(final_rhs)
        return solution_vector


class LeapFrogIntegrator(Integrator):
    def __init__(self, solution_vector, parameters):
        super(LeapFrogIntegrator, self).__init__(solution_vector, parameters)

    def integrate(self, solution_vector):
        dummy = solution_vector
        intermediate_rhs = self.fluxer.calculate_rhs(solution_vector)
        self.lagged_solution.update(2.0 * intermediate_rhs)
        solution_vector = self.lagged_solution
        self.lagged_solution = dummy
        return solution_vector


class RK2Integrator(Integrator):
    def __init__(self, solution_vector, parameters):
        super(RK2Integrator, self).__init__(solution_vector, parameters)

    def integrate(self, solution_vector):
        k1 = self.fluxer.calculate_rhs(solution_vector)
        solution_vector.update(0.75 * k1)
        k2 = self.fluxer.calculate_rhs(solution_vector)
        solution_vector.update(0.333 * k1 + 0.666 * k2)
        return solution_vector
