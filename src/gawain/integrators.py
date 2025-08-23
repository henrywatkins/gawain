"""Integrators

These integrators take a solution vector and integrate
(update) the solution depending on some RHS; in practice
this means taking the fluxes of the variables in and out
of a cell and updating the cell accordingly.
"""

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .io import Parameters
    from .numerics import GravitySource, MHDSolutionVector, SolutionVector


class Integrator:
    """Integration class

    Integrate the solution given the RHS of the equation,
    timestep size and any sources.

    Attributes
    ----------
    fluxer : FluxCalculator-type object
        used to calcualte the equation RHS
    source : ndarray
        used to calculate contributions from arbitrary sources
    gravity : GravitySource object
        contains informaiton about gravitational field
    """

    def __init__(self, parameters: "Parameters") -> None:
        self.fluxer = parameters.create_fluxer()
        self.fluxer.set_flux_function(parameters.with_mhd)
        self.source = parameters.create_source()
        self.gravity = parameters.create_gravity()

    def integrate(
        self, solution_vector: "Union[SolutionVector, MHDSolutionVector]"
    ) -> "Union[SolutionVector, MHDSolutionVector]":
        """Integrate the solution at this timestep

        Parameters
        ----------
        solution_vector : SolutionVector-type object
            a solution vector to be integrated at this timestep

        Returns
        -------
        solution_vector : SolutionVector-type object
            the updated solution
        """
        intermediate_rhs = self.fluxer.calculate_flux_divergence(solution_vector)
        if self.source is not None:
            intermediate_rhs += self.source
        if self.gravity is not None:
            intermediate_rhs += self.gravity.calculate_gravity_source(solution_vector)
        solution_vector.update(intermediate_rhs)
        return solution_vector
