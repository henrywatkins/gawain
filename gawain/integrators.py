""" integrator classes
"""

import numpy as np


class Integrator:
    def __init__(self, SolutionVector, Parameters):
        self.lagged_solution = SolutionVector
        self.fluxer = Parameters.fluxer_type()

    def integrate(self, SolutionVector):
        intermediate_rhs = self.fluxer.calculate_rhs(SolutionVector)
        SolutionVector.update(intermediate_rhs)
        return SolutionVector


class PredictorCorrectorIntegrator(Integrator):
    def __init__(self, SolutionVector, Parameters):
        super(PredictorCorrectorIntegrator, self).__init__(SolutionVector, Parameters)

    def integrate(self, SolutionVector):
        intermediate_rhs = self.fluxer.calculate_rhs(SolutionVector)
        intermediate_solution = SolutionVector
        intermediate_solution.update(intermediate_rhs)
        final_rhs = self.fluxer.calculate_rhs(intermediate_solution)
        final_rhs = 0.5*(intermediate_rhs + final_rhs)
        SolutionVector.update(final_rhs)
        return SolutionVector

class LeapFrogIntegrator(Integrator):
    def __init__(self, SolutionVector, Parameters):
        super(LeapFrogIntegrator, self).__init__(SolutionVector, Parameters)

    def integrate(self, SolutionVector):
        dummy = SolutionVector
        intermediate_rhs = self.fluxer.calculate_rhs(SolutionVector)
        self.lagged_solution.update(2.*intermediate_rhs)
        SolutionVector = self.lagged_solution
        self.lagged_solution = dummy
        return SolutionVector

class RK2Integrator(Integrator):
    def __init__(self, SolutionVector, Parameters):
        super(RK2Integrator, self).__init__(SolutionVector, Parameters)

    def integrate(self, SolutionVector):
        k1 = self.fluxer.calculate_rhs(SolutionVector)
        SolutionVector.update(0.75*k1)
        k2 = self.fluxer.calculate_rhs(SolutionVector)
        SolutionVector.update(0.333*k1 + 0.666*k2)
        return SolutionVector