''' Numerical utilities '''

import time

from tqdm import tqdm
import numpy as np

class Clock:
    def __init__(self, Parameters):
        self.current_time = 0.0
        self.end_time = Parameters.t_max
        self.cfl = Parameters.cfl
        self.timestep = 0.0001
        self.next_output_time = 0.0
        self.output_spacing = self.end_time/Parameters.n_outputs
        self.bar = tqdm(total=self.end_time)
        self.wallclock_start = time.process_time()

    def is_end(self):
        if self.current_time<self.end_time:
            return False
        else:
            return True

    def tick(self):
        self.bar.update(self.timestep)
        self.current_time += self.timestep

    def is_output(self):
        if self.current_time >= self.next_output_time:
            self.next_output_time += self.output_spacing
            return True
        else:
            return False


    def duration(self):
        wallclock_end = time.process_time()
        return wallclock_end - self.wallclock_start


    def calculate_timestep(self, SolutionVector):
        return self.timestep


class SolutionVector:
    def __init__(self, Parameters):
        self.data = None
        self.boundary_type = Parameters.boundary_type
        self.boundary_value = Parameters.boundary_value
        self.adi_idx = Parameters.adi_idx
        self.set_centroid(Parameters.initial_condition)

    def set_centroid(self, array):
        self.data = array

    def centroid(self):
        return self.data

    def plusX(self, n=1):
        rolled = np.roll(self.data, n, axis=1)
        if self.boundary_type[0]=="periodic":
            return rolled
        elif self.boundary_type[0]=="fixed":
            rolled[:,0] = self.boundary_value[0][0]
            return rolled

    def minusX(self, n=1):
        rolled = np.roll(self.data, -n, axis=1)
        if self.boundary_type[0]=="periodic":
            return rolled
        elif self.boundary_type[0]=="fixed":
            rolled[:,-1] = self.boundary_value[0][1]
            return rolled

    def plusY(self, n=1):
        rolled = np.roll(self.data, n, axis=2)
        if self.boundary_type[1]=="periodic":
            return rolled
        elif self.boundary_type[1]=="fixed":
            rolled[:,:,0] = self.boundary_value[1][0]
            return rolled

    def minusY(self, n=1):
        rolled = np.roll(self.data, -n, axis=2)
        if self.boundary_type[1]=="periodic":
            return rolled
        elif self.boundary_type[1]=="fixed":
            rolled[:,:,-1] = self.boundary_value[1][1]
            return rolled
    
    def plusZ(self, n=1):
        rolled = np.roll(self.data, n, axis=3)
        if self.boundary_type[2]=="periodic":
            return rolled
        elif self.boundary_type[2]=="fixed":
            rolled[:,:,0] = self.boundary_value[2][0]
            return rolled

    def minusZ(self, n=1):
        rolled = np.roll(self.data, -n, axis=3)
        if self.boundary_type[2]=="periodic":
            return rolled
        elif self.boundary_type[2]=="fixed":
            rolled[:,:,-1] = self.boundary_value[2][1]
            return rolled

    def update(self, array):
        self.data+=array

    def adi_minus1(self):
        return self.adi_idx-1

    def dens(self):
        return self.data[0]
    def momX(self):
        return self.data[1]
    def momY(self):
        return self.data[2]
    def momZ(self):
        return self.data[3]
    def momMagSqr(self):
        return self.data[1]*self.data[1]+self.data[2]*self.data[2]+self.data[3]*self.data[3]
    def en(self):
        return self.data[4]
    def sound_speed(self):
        pass

class Integrator:
    def __init__(self, SolutionVector, Parameters):
        self.lagged_solution = SolutionVector
        self.fluxer = Parameters.fluxer_type(Parameters)

    def integrate(self, SolutionVector, time_step):
        intermediate_rhs = self.fluxer.calculate_rhs(SolutionVector, time_step)
        SolutionVector.update(time_step*intermediate_rhs)
        return SolutionVector


class PredictorCorrectorIntegrator(Integrator):
    def __init__(self, SolutionVector, Parameters):
        super(PredictorCorrectorIntegrator, self).__init__(SolutionVector, Parameters)

    def integrate(self, SolutionVector, time_step):
        intermediate_rhs = self.fluxer.calculate_rhs(SolutionVector, time_step)
        intermediate_solution = SolutionVector
        intermediate_solution.update(time_step*intermediate_rhs)
        final_rhs = self.fluxer.calculate_rhs(intermediate_solution, time_step)
        final_rhs = 0.5*(intermediate_rhs + final_rhs)
        SolutionVector.update(time_step*final_rhs)
        return SolutionVector

class LeapFrogIntegrator(Integrator):
    def __init__(self, SolutionVector, Parameters):
        super(LeapFrogIntegrator, self).__init__(SolutionVector, Parameters)

    def integrate(self, SolutionVector, time_step):
        dummy = SolutionVector
        intermediate_rhs = self.fluxer.calculate_rhs(SolutionVector, time_step)
        self.lagged_solution.update(2.*time_step*intermediate_rhs)
        SolutionVector = self.lagged_solution
        self.lagged_solution = dummy
        return SolutionVector

class RK2Integrator(Integrator):
    def __init__(self, SolutionVector, Parameters):
        super(RK2Integrator, self).__init__(SolutionVector, Parameters)

    def integrate(self, SolutionVector, time_step):
        k1 = self.fluxer.calculate_rhs(SolutionVector, time_step)
        k1 *= time_step
        SolutionVector.update(0.75*k1)
        k2 = self.fluxer.calculate_rhs(SolutionVector, time_step)
        k2 *= time_step
        SolutionVector.update(0.333*k1 + 0.666*k2)
        return SolutionVector
