'''
        GAWAIN

    A small python R-MHD code for
    education and experimentation

'''

import sys
import os
import time



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

    start_time =  time.process_time()

    input_file = str(sys.argv[1])
    params = io.Parameters(input_file)
    params.print_params()
    clock = nu.Clock(params)
    solution = nu.SolutionVector(params)

    while not clock.is_end():
        dt = clock.calculate_timestep(solution)
        solution.calculate_fluxes()
        solution.update(dt)

        if clock.is_output():
            io.output_data(solution)

        clock.update()

    end_time = time.process_time()

    print('\nSimulation Complete, duration:', end_time-start_time, 'seconds')



if __name__ == "__main__":
    main()
