# Gawain notes

## Boundary conditions

Use boundary setter

1. given axis and direction, get boundary cell indices
   - 
2. calculate what those values should be for this boundary type
3. replace the boundary cells using with these new values





## Old integrators

```python
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
        self.lagged_solution = solution_vector

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
```

## Testing with MHD

linear waves look like they are working fine but brio-wu v. problematic - is this to do with HLL solver or with mag field divergence? Possible calculation of wave speeds

- brio wu test divergence is zero
- HLL solver works fine for SOD tube, 
- error still appears with fixed small timestep, not due to timestep calculation
- possibly due to pressure/mag pressure calculation in mhdsolutionvector

Problem was due to errors in MHD flux function calculation and in the magnetosonic wave speed calculation

- MHD rotor keeps coming up against NAN values even though none exist in inititial condition

## Testing Hydro

- In both the Rayleigh-Taylor and Kelvin-Helmholtz instability runs, the numerical diffusion is too large and damping out instability seed before they can grow. HLLE is not sufficient for these instabilities and converting to HLLC will be necessary.
- The run time does not appear to grow linearly with mesh size - analysis of scaling is required - some profiling necessary too
- during interatcting blast wave, get singularity in timestep calculation when the two waves interact

## Future development

- GPU support
- HLLD solver
- 3D support
- type hints
- resistivity
- thermal conductivity