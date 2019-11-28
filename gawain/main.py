'''
        GAWAIN

    A small python R-MHD code for
    education and experimentation

'''

import sys
import os
import time

import progressbar
import gawain.io as io
import gawain.numerics as nu


preamble =


'  ______                     _      \n
  / ____/___ __      ______ _(_)___  \n
 / / __/ __ `/ | /| / / __ `/ / __ \ \n
/ /_/ / /_/ /| |/ |/ / /_/ / / / / / \n
\____/\__,_/ |__/|__/\__,_/_/_/ /_/  \n'




def main():

    print(preamble)

    input_file = str(sys.argv[1])
    params = io.Parameters(input_file)
    clock = nu.Clock(params)
    solution = nu.SolutionVector(params)

    while not clock.is_end():
        dt = clock.calculate_timestep()
        solution.calculate_fluxes()
        solution.update(dt)

        if clock.is_output():
            io.output_data(solution)

        clock.update()

    print('Simulation Complete')



if __name__ == "__main__":
    main()
