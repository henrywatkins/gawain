"""Integrators 

These integrators take a solution vector and integrate
(update) the solution depending on some RHS; in practice
this means taking the fluxes of the variables in and out
of a cell and updating the cell accordingly.
"""

import numpy as np


class Integrator:
    def __init__(self, parameters):
        self.fluxer = parameters.create_fluxer()
        self.fluxer.set_flux_function(parameters.with_mhd)
        self.source = parameters.create_source()

    def integrate(self, solution_vector):
        intermediate_rhs = self.fluxer.calculate_rhs(solution_vector)
        if self.source is not None:
            intermediate_rhs += self.source
        solution_vector.update(intermediate_rhs)
        return solution_vector
