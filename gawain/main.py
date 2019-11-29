'''
        GAWAIN

    A small python R-MHD code for
    education and experimentation

'''

import sys

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


def main():

    print(preamble)

    input_file = str(sys.argv[1])
    params = io.Parameters(input_file)
    params.print_params()
    clock = nu.Clock(params)
    solution = nu.SolutionVector(params)
    output = io.Output(params, solution)

    while not clock.is_end():
        dt = clock.calculate_timestep(solution)
        solution.update(dt)

        if clock.is_output():
            output.dump(solution)

        clock.tick()


    print('\nSimulation Complete, duration:', clock.duration(), 'seconds')



if __name__ == "__main__":
    main()
