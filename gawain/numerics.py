""" Numerical utilities """

import time

from tqdm import tqdm
import numpy as np


class Clock:
    def __init__(self, Parameters):
        self.current_time = 0.0
        self.end_time = Parameters.t_max
        self.next_output_time = 0.0
        self.output_spacing = self.end_time / Parameters.n_outputs
        self.bar = tqdm(total=self.end_time + 0.01)
        self.wallclock_start = time.process_time()

    def is_end(self):
        if self.current_time < self.end_time:
            return False
        else:
            self.bar.close()
            return True

    def tick(self, dt):
        self.bar.update(dt)
        self.current_time += dt

    def is_output(self):
        if self.current_time >= self.next_output_time:
            self.next_output_time += self.output_spacing
            return True
        else:
            return False

    def duration(self):
        wallclock_end = time.process_time()
        return wallclock_end - self.wallclock_start


class SolutionVector:
    def __init__(self):
        self.data = None
        self.boundary_type = None
        self.boundary_value = None
        self.dx, self.dy, self.dz = None, None, None
        self.adi_idx = 1.4
        self.timestep = 0.0001
        self.cfl = 0.1
        self.variable_names = [
            "density",
            "xmomentum",
            "ymomentum",
            "zmomentum",
            "energy",
        ]

    def set_state(self, Parameters):
        self.boundary_type = Parameters.boundary_type
        self.boundary_value = Parameters.boundary_value
        self.dx, self.dy, self.dz = Parameters.cell_sizes
        self.adi_idx = Parameters.adi_idx
        self.set_centroid(Parameters.initial_condition)
        self.cfl = Parameters.cfl

    def copy(self):
        new_vector = SolutionVector()
        new_vector.data = self.data
        new_vector.boundary_type = self.boundary_type
        new_vector.boundary_value = self.boundary_value
        new_vector.dx, new_vector.dy, new_vector.dz = self.dx, self.dy, self.dz
        new_vector.adi_idx = self.adi_idx
        new_vector.timestep = self.timestep
        return new_vector

    def calculate_timestep(self):
        cs_max = self.sound_speed().max()
        timestep_x = self.cfl * self.dx / cs_max
        timestep_y = self.cfl * self.dy / cs_max
        self.timestep = min(timestep_x, timestep_y)
        return self.timestep

    def set_centroid(self, array):
        self.data = array

    def centroid(self):
        return self.data

    def get_variable(self, variable_name):
        index = self.variable_names.index(variable_name)
        return self.data[index]

    def plusX(self, n=1):
        rolled = np.roll(self.data, n, axis=1)
        if self.boundary_type[0] == "periodic":
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[0] == "fixed":
            rolled[:, 0] = self.boundary_value[0][0]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[0] == "reflective":
            rolled[0, 0] = self.data[0, 0]
            rolled[1, 0] = -self.data[1, 0]
            rolled[4, 0] = self.data[4, 0]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector

    def minusX(self, n=1):
        rolled = np.roll(self.data, -n, axis=1)
        if self.boundary_type[0] == "periodic":
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[0] == "fixed":
            rolled[:, -1] = self.boundary_value[0][1]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[0] == "reflective":
            rolled[0, -1] = self.data[0, -1]
            rolled[1, -1] = -self.data[1, -1]
            rolled[4, -1] = self.data[4, -1]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector

    def plusY(self, n=1):
        rolled = np.roll(self.data, n, axis=2)
        if self.boundary_type[1] == "periodic":
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[1] == "fixed":
            rolled[:, :, 0] = self.boundary_value[1][0]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[1] == "reflective":
            rolled[0, :, 0] = self.data[0, :, 0]
            rolled[1, :, 0] = -self.data[1, :, 0]
            rolled[4, :, 0] = self.data[4, :, 0]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector

    def minusY(self, n=1):
        rolled = np.roll(self.data, -n, axis=2)
        if self.boundary_type[1] == "periodic":
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[1] == "fixed":
            rolled[:, :, -1] = self.boundary_value[1][1]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[1] == "reflective":
            rolled[0, :, -1] = self.data[0, :, -1]
            rolled[1, :, -1] = -self.data[1, :, -1]
            rolled[4, :, -1] = self.data[4, :, -1]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector

    def plusZ(self, n=1):
        rolled = np.roll(self.data, n, axis=3)
        if self.boundary_type[2] == "periodic":
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[2] == "fixed":
            rolled[:, :, :, 0] = self.boundary_value[2][0]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[2] == "reflective":
            rolled[0, :, :, 0] = self.data[0, :, :, 0]
            rolled[1, :, :, 0] = -self.data[1, :, :, 0]
            rolled[4, :, :, 0] = self.data[4, :, :, 0]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector

    def minusZ(self, n=1):
        rolled = np.roll(self.data, -n, axis=3)
        if self.boundary_type[2] == "periodic":
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[2] == "fixed":
            rolled[:, :, :, -1] = self.boundary_value[2][1]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector
        elif self.boundary_type[2] == "reflective":
            rolled[0, :, :, -1] = self.data[0, :, :, -1]
            rolled[1, :, :, -1] = -self.data[1, :, :, -1]
            rolled[4, :, :, -1] = self.data[4, :, :, -1]
            new_vector = self.copy()
            new_vector.set_centroid(rolled)
            return new_vector

    def update(self, array):
        self.data += self.timestep * array

    def dens(self):
        return self.data[0]

    def momX(self):
        return self.data[1]

    def momY(self):
        return self.data[2]

    def momZ(self):
        return self.data[3]

    def velX(self):
        return self.data[1] / self.data[0]

    def velY(self):
        return self.data[2] / self.data[0]

    def velZ(self):
        return self.data[3] / self.data[0]

    def momTotalSqr(self):
        return (
            self.data[1] * self.data[1]
            + self.data[2] * self.data[2]
            + self.data[3] * self.data[3]
        )

    def energy(self):
        return self.data[4]

    def pressure(self):
        adi_minus1 = self.adi_idx - 1.0
        thermal_en = self.energy() - 0.5 * self.momTotalSqr() / self.dens()
        pressure = adi_minus1 * thermal_en
        return pressure

    def sound_speed(self):
        return np.sqrt(self.adi_idx * self.pressure() / self.dens())

    # methods for extracting data for vector
    # data along each direction
    def mom(self, dim):
        return self.data[dim + 1]

    def vel(self, dim):
        return self.data[dim + 1] / self.data[0]

    def plus_shift(self, dim):
        rolled = np.roll(self.data, 1, axis=dim + 1)
        if self.boundary_type[dim] == "periodic":
            pass
        elif self.boundary_type[dim] == "fixed":
            rolled[:, :, :, 0] = self.boundary_value[dim][0]
        elif self.boundary_type[dim] == "reflective":
            rolled[0, :, :, 0] = self.data[0, :, :, 0]
            rolled[1, :, :, 0] = -self.data[1, :, :, 0]
            rolled[4, :, :, 0] = self.data[4, :, :, 0]

        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def minus_shift(self, dim):
        rolled = np.roll(self.data, -1, axis=dim + 1)
        if self.boundary_type[dim] == "periodic":
            pass
        elif self.boundary_type[dim] == "fixed":
            rolled[:, :, :, -1] = self.boundary_value[dim][1]
        elif self.boundary_type[dim] == "reflective":
            rolled[0, :, :, -1] = self.data[0, :, :, -1]
            rolled[1, :, :, -1] = -self.data[1, :, :, -1]
            rolled[4, :, :, -1] = self.data[4, :, :, -1]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector


class MHDSolutionVector(SolutionVector):
    def __init__(self):
        super(MHDSolutionVector, self).__init__()
        self.variable_names = [
            "density",
            "xmomentum",
            "ymomentum",
            "zmomentum",
            "energy",
            "xmag",
            "ymag",
            "zmag",
        ]

    def magX(self):
        return self.data[5]

    def magY(self):
        return self.data[6]

    def magZ(self):
        return self.data[7]

    def magTotalSqr(self):
        return (
            self.data[5] * self.data[5]
            + self.data[6] * self.data[6]
            + self.data[7] * self.data[7]
        )

    def magnetic_pressure(self):
        return self.magTotalSqr() / 2.0

    def pressure(self):
        adi_minus1 = self.adi_idx - 1.0
        thermal_en = (
            self.energy()
            - 0.5 * self.momTotalSqr() / self.dens()
            - self.magnetic_pressure()
        )
        pressure = adi_minus1 * thermal_en
        return pressure

    def alfven_speed(self):
        return np.sqrt(self.magTotalSqr() / self.dens())

    def fast_magnetosonic_speed(self):
        gamma_pressure = self.adi_idx * self.pressure()
        b = gamma_pressure + self.magTotalSqr()
        in_x = (
            0.5
            * (b + np.sqrt(b ** 2 - 4 * gamma_pressure * self.magX() * self.magX()))
            / self.dens()
        )
        in_y = (
            0.5
            * (b + np.sqrt(b ** 2 - 4 * gamma_pressure * self.magY() * self.magY()))
            / self.dens()
        )
        in_z = (
            0.5
            * (b + np.sqrt(b ** 2 - 4 * gamma_pressure * self.magZ() * self.magZ()))
            / self.dens()
        )
        return np.array([in_x, in_y, in_z])

    def slow_magnetosonic_speed(self):
        gamma_pressure = self.adi_idx * self.pressure()
        b = gamma_pressure + self.magTotalSqr()
        in_x = (
            0.5
            * (b - np.sqrt(b ** 2 - 4 * gamma_pressure * self.magX() * self.magX()))
            / self.dens()
        )
        in_y = (
            0.5
            * (b - np.sqrt(b ** 2 - 4 * gamma_pressure * self.magY() * self.magY()))
            / self.dens()
        )
        in_z = (
            0.5
            * (b - np.sqrt(b ** 2 - 4 * gamma_pressure * self.magZ() * self.magZ()))
            / self.dens()
        )
        return np.array([in_x, in_y, in_z])

    def calculate_timestep(self):
        ca_max = self.alfven_speed().max()
        cs = self.slow_magnetosonic_speed()
        cf = self.fast_magnetosonic_speed()
        csx_max, csy_max = np.abs(cs[0]).max(), np.abs(cs[1]).max()
        cfx_max, cfy_max = np.abs(cf[0]).max(), np.abs(cf[1]).max()
        cx_max = max(ca_max, csx_max, cfx_max)
        cy_max = max(ca_max, csy_max, cfy_max)
        timestep_x = self.dx / cx_max
        timestep_y = self.dy / cy_max
        self.timestep = min(timestep_x, timestep_y)
        return self.cfl * self.timestep
