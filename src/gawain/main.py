"""GAWAIN

A small ideal-MHD code for trying out
hydrodynamics and magetohydrodynamics
in python.

"""

import gawain.io as io
import gawain.numerics as nu

PREAMBLE = """
   ______                     _
  / ____/___ __      ______ _(_)___
 / / __/ __ `/ | /| / / __ `/ / __ |
/ /_/ / /_/ /| |/ |/ / /_/ / / / / /
|____/|__,_/ |__/|__/|__,_/_/_/ /_/
-----------------------------------
        MHD simulation code
-----------------------------------
Simulation parameters:
"""


def run_gawain(config):

    print(PREAMBLE)

    params = io.Parameters(config)
    if params.with_mhd:
        solution = nu.MHDSolutionVector()
    else:
        solution = nu.SolutionVector()
    solution.set_state(params)
    integrator = params.create_integrator()
    params.print_params()
    output = io.Output(params, solution)
    clock = nu.Clock(params)

    while not clock.is_end():
        dt = solution.calculate_timestep()
        solution = integrator.integrate(solution)

        if clock.is_output():
            output.dump(solution)

        clock.tick(dt)
        
    clock.dump_times(output.save_dir)

    print(f"\nSimulation Complete, duration: {clock.duration():.2f} seconds")
