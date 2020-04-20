'''
        GAWAIN

    A small python R-MHD code for
    education and experimentation

'''

import sys
import plac

import gawain.io as io
import gawain.numerics as nu


preamble = """
   ______                     _
  / ____/___ __      ______ _(_)___
 / / __/ __ `/ | /| / / __ `/ / __ |
/ /_/ / /_/ /| |/ |/ / /_/ / / / / /
\____/\__,_/ |__/|__/\__,_/_/_/ /_/
-----------------------------------
      R-MHD simulation code
-----------------------------------
Simulation parameters:
"""

@plac.annotations(
        input_file=("Input runfile path", "positional", None, str),
        output_path=("Output folder path", "option", "o", str),
)


def main(input_file, output_path=None):

    print(preamble)

    params = io.Parameters(from_file=input_file)
    solution = nu.SolutionVector()
    solution.set_state(params)
    integrator = params.integrator_type(solution, params)
    params.print_params()
    output = io.Output(params, solution)
    clock = nu.Clock(params)

    while not clock.is_end():
        dt = solution.calculate_timestep()
        solution = integrator.integrate(solution)

        if clock.is_output():
            output.dump(solution)

        clock.tick(dt)


    print('\nSimulation Complete, duration:', clock.duration(), 'seconds')



if __name__ == "__main__":
    plac.call(main)
