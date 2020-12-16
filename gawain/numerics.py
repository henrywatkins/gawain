"""Numerical utilities

These numerical utilities include the Clock
time-keeping class and the base solution vector
classes for both hydro and mhd.
"""

import time

from tqdm import tqdm
import numpy as np


class Clock:
    """Timing utility for simulations
    
    This clock timer keeps track of the simulation run time
    
    Attributes
    ----------
    current_time : float
        the current simulation time 
    end_time : float
        the time the simulation will end
    next_outout_time : float
        the next time point at which output will be dumped
    """

    def __init__(self, Parameters):
        """
        Parameters
        ----------
        Parameters : a Parameters object
            an object of the Parameters class holding simulation configuration
        """
        self.current_time = 0.0
        self.end_time = Parameters.t_max
        self.next_output_time = 0.0
        self.output_spacing = self.end_time / Parameters.n_outputs
        self.bar = tqdm(total=self.end_time + 0.01)
        self.wallclock_start = time.process_time()

    def is_end(self):
        """Check if simulation has reached the end time"""

        if self.current_time < self.end_time:
            return False
        else:
            self.bar.close()
            return True

    def tick(self, dt):
        """update the current time"""
        self.bar.update(dt)
        self.current_time += dt

    def is_output(self):
        """check if output should be dumped at this timestep"""
        if self.current_time >= self.next_output_time:
            self.next_output_time += self.output_spacing
            return True
        else:
            return False

    def duration(self):
        """calculate the total duration of the simulation in seconds"""
        wallclock_end = time.process_time()
        return wallclock_end - self.wallclock_start


class SolutionVector:
    """Hydrodynamic solution vector field
    
    This solution vector contains all vector field data for 
    all variables in the hydrodynamic problem.
    
    Attributes
    ----------
    data : ndarray
        the raw solution data, of shape (n_variables, Nx, Ny, Nz)
    boundary_type : List[str]
        the boundary condition type in the x, y ,z axes
    boundary_value : List[float]
        the value of the boundary is using fixed boundary condition
    dx, dy, dz : float
        the cell widths in x, y , z directions
    cfl : float
        the value of the CFL condition parameter
    adi_idx : float
        the adiabatic index
    timestep : float
        the current timestep size 
    variable_names : List[str]
        the names of the variables contained in the solution vector, in order of their vecotr position
    """

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
        """Set the initial state of the solution vector
        
        Parameters
        ----------
        Parameters : a Parameters object
            the simulation configuration object
        """
        self.boundary_type = Parameters.boundary_type
        self.boundary_value = Parameters.boundary_value
        self.dx, self.dy, self.dz = Parameters.cell_sizes
        self.adi_idx = Parameters.adi_idx
        self.set_centroid(Parameters.initial_condition)
        self.cfl = Parameters.cfl

    def copy(self):
        """Return a copy of the solution vector"""
        new_vector = SolutionVector()
        new_vector.data = self.data
        new_vector.boundary_type = self.boundary_type
        new_vector.boundary_value = self.boundary_value
        new_vector.dx, new_vector.dy, new_vector.dz = self.dx, self.dy, self.dz
        new_vector.adi_idx = self.adi_idx
        new_vector.timestep = self.timestep
        return new_vector

    def calculate_min_max_wave_speeds_X(self):
        """Return the the minimum and maximum wave speeds in the X direction
        
        Returns
        -------
        Tuple[ndarray, ndarray]
            the minimum and maximum wave speeds in the x direction for each cell
        """
        xvel = self.velX()
        cs = self.sound_speed()
        lambda1 = xvel - cs
        lambda2 = xvel + cs
        return np.minimum(lambda1, lambda2), np.maximum(lambda1, lambda2)

    def calculate_min_max_wave_speeds_Y(self):
        """Return the the minimum and maximum wave speeds in the Y direction
        
        Returns
        -------
        Tuple[ndarray, ndarray]
            the minimum and maximum wave speeds in the y direction for each cell
        """
        yvel = self.velY()
        cs = self.sound_speed()
        lambda1 = yvel - cs
        lambda2 = yvel + cs
        return np.minimum(lambda1, lambda2), np.maximum(lambda1, lambda2)

    def calculate_timestep(self):
        """Calculate the timestep size, using the wave speeds and CFL value
        
        Returns
        -------
        timestep : float
            the new timestep size dt
        """
        min_wave_speed_x, max_wave_speed_x = self.calculate_min_max_wave_speeds_X()
        min_wave_speed_y, max_wave_speed_y = self.calculate_min_max_wave_speeds_Y()
        max_in_x = max(np.abs(min_wave_speed_x).max(), np.abs(max_wave_speed_x).max())
        max_in_y = max(np.abs(min_wave_speed_y).max(), np.abs(max_wave_speed_y).max())
        timestep_x = self.cfl * self.dx / max_in_x
        timestep_y = self.cfl * self.dy / max_in_y
        self.timestep = min(timestep_x, timestep_y)
        return self.timestep

    def set_centroid(self, array):
        """Set the data for each cell in the mesh"""
        self.data = array

    def centroid(self):
        """Return the solution vector data for each mesh cell
        
        Returns
        -------
        ndarray
            solution vector data for all variables
        """
        return self.data

    def get_variable(self, variable_name):
        """Get the solution data specific to a variable
        
        Parameters
        ----------
        variable_name : str
            the name of the variable data to return
        
        Returns
        -------
        ndarray
            the solution vector data for the specified variable
        """
        index = self.variable_names.index(variable_name)
        return self.data[index]

    def plusX(self):
        """Shift all the solution data in the positive x direction
        
        This operation provides the means to acces the +x position of the 
        numerical stencil in a vecotrized manner. It makes a copy of the
        solution vector with the data shifted to required stencil position, 
        accounting for boundary conditions.
        
        Returns
        -------
        new_vector : SolutionVector
            the +X solution vector for every cell in the mesh
        """
        rolled = np.roll(self.data, 1, axis=1)
        if self.boundary_type[0] == "periodic":
            pass
        elif self.boundary_type[0] == "fixed":
            rolled[:, 0] = self.boundary_value[0][0]
        elif self.boundary_type[0] == "reflective":
            rolled[0, 0] = self.data[0, 0]
            rolled[1, 0] = -self.data[1, 0]
            rolled[4, 0] = self.data[4, 0]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def minusX(self):
        """Shift all the solution data in the negative x direction
        
        This operation provides the means to acces the -x position of the 
        numerical stencil in a vecotrized manner. It makes a copy of the
        solution vector with the data shifted to required stencil position, 
        accounting for boundary conditions.
        
        Returns
        -------
        new_vector : SolutionVector
            the -X solution vector for every cell in the mesh
        """
        rolled = np.roll(self.data, -1, axis=1)
        if self.boundary_type[0] == "periodic":
            pass
        elif self.boundary_type[0] == "fixed":
            rolled[:, -1] = self.boundary_value[0][1]
        elif self.boundary_type[0] == "reflective":
            rolled[0, -1] = self.data[0, -1]
            rolled[1, -1] = -self.data[1, -1]
            rolled[4, -1] = self.data[4, -1]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def plusY(self):
        """Shift all the solution data in the positive y direction
        
        This operation provides the means to acces the +y position of the 
        numerical stencil in a vecotrized manner. It makes a copy of the
        solution vector with the data shifted to required stencil position, 
        accounting for boundary conditions.
        
        Returns
        -------
        new_vector : SolutionVector
            the +y solution vector for every cell in the mesh
        """
        rolled = np.roll(self.data, 1, axis=2)
        if self.boundary_type[1] == "periodic":
            pass
        elif self.boundary_type[1] == "fixed":
            rolled[:, :, 0] = self.boundary_value[1][0]
        elif self.boundary_type[1] == "reflective":
            rolled[0, :, 0] = self.data[0, :, 0]
            rolled[1, :, 0] = -self.data[1, :, 0]
            rolled[4, :, 0] = self.data[4, :, 0]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def minusY(self):
        """Shift all the solution data in the negative y direction
        
        This operation provides the means to acces the -y position of the 
        numerical stencil in a vecotrized manner. It makes a copy of the
        solution vector with the data shifted to required stencil position, 
        accounting for boundary conditions.
        
        Returns
        -------
        new_vector : SolutionVector
            the -y solution vector for every cell in the mesh
        """
        rolled = np.roll(self.data, -1, axis=2)
        if self.boundary_type[1] == "periodic":
            pass
        elif self.boundary_type[1] == "fixed":
            rolled[:, :, -1] = self.boundary_value[1][1]
        elif self.boundary_type[1] == "reflective":
            rolled[0, :, -1] = self.data[0, :, -1]
            rolled[1, :, -1] = -self.data[1, :, -1]
            rolled[4, :, -1] = self.data[4, :, -1]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def plusZ(self):
        """Shift all the solution data in the positive z direction
        
        This operation provides the means to acces the +z position of the 
        numerical stencil in a vecotrized manner. It makes a copy of the
        solution vector with the data shifted to required stencil position, 
        accounting for boundary conditions.
        
        Returns
        -------
        new_vector : SolutionVector
            the +z solution vector for every cell in the mesh
        """
        rolled = np.roll(self.data, 1, axis=3)
        if self.boundary_type[2] == "periodic":
            pass
        elif self.boundary_type[2] == "fixed":
            rolled[:, :, :, 0] = self.boundary_value[2][0]
        elif self.boundary_type[2] == "reflective":
            rolled[0, :, :, 0] = self.data[0, :, :, 0]
            rolled[1, :, :, 0] = -self.data[1, :, :, 0]
            rolled[4, :, :, 0] = self.data[4, :, :, 0]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def minusZ(self):
        """Shift all the solution data in the neagtive z direction
        
        This operation provides the means to acces the -z position of the 
        numerical stencil in a vecotrized manner. It makes a copy of the
        solution vector with the data shifted to required stencil position, 
        accounting for boundary conditions.
        
        Returns
        -------
        new_vector : SolutionVector
            the -z solution vector for every cell in the mesh
        """
        rolled = np.roll(self.data, -1, axis=3)
        if self.boundary_type[2] == "periodic":
            pass
        elif self.boundary_type[2] == "fixed":
            rolled[:, :, :, -1] = self.boundary_value[2][1]
        elif self.boundary_type[2] == "reflective":
            rolled[0, :, :, -1] = self.data[0, :, :, -1]
            rolled[1, :, :, -1] = -self.data[1, :, :, -1]
            rolled[4, :, :, -1] = self.data[4, :, :, -1]
        new_vector = self.copy()
        new_vector.set_centroid(rolled)
        return new_vector

    def update(self, array):
        """Update the solution vector data with an array of values
        
        u' = u + delta t * array
        
        Parameters
        ----------
        array : ndarray
            the array to update the solution vector
        """

        self.data += self.timestep * array

    def dens(self):
        """Return the density field data"""
        return self.data[0]

    def momX(self):
        """Return the x-momentumm field data"""
        return self.data[1]

    def momY(self):
        """Return the y-momentum field data"""
        return self.data[2]

    def momZ(self):
        """Return the z-momentum field data"""
        return self.data[3]

    def velX(self):
        """Return the x-velocity field data"""
        return self.data[1] / self.data[0]

    def velY(self):
        """Return the y-velocity field data"""
        return self.data[2] / self.data[0]

    def velZ(self):
        """Return the z-velocity field data"""
        return self.data[3] / self.data[0]

    def momTotalSqr(self):
        """Return the total momentum squared field data |M|**2"""
        return self.data[1] ** 2 + self.data[2] ** 2 + self.data[3] ** 2

    def energy(self):
        """Return the total energy field data"""
        return self.data[4]

    def pressure(self):
        """Return the thermal pressure field data"""
        thermal_en = self.energy() - 0.5 * self.momTotalSqr() / self.dens()
        pressure = (self.adi_idx - 1.0) * thermal_en
        return pressure

    def sound_speed(self):
        """Return the sound speed for every mesh cell"""
        return np.sqrt(self.adi_idx * self.pressure() / self.dens())


class MHDSolutionVector(SolutionVector):
    """Magnetohydrodynamic solution vector field
    
    This solution vector contains all vector field data for 
    all variables in the magnetohydrodynamic problem.
    """

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

    def copy(self):
        """Return a copy of the solution vector"""
        new_vector = MHDSolutionVector()
        new_vector.data = self.data
        new_vector.boundary_type = self.boundary_type
        new_vector.boundary_value = self.boundary_value
        new_vector.dx, new_vector.dy, new_vector.dz = self.dx, self.dy, self.dz
        new_vector.adi_idx = self.adi_idx
        new_vector.timestep = self.timestep
        return new_vector

    def magX(self):
        """Return the x-direction magnetic field data"""
        return self.data[5]

    def magY(self):
        """Return the y-direction magnetic field data"""
        return self.data[6]

    def magZ(self):
        """Return the z-direction magnetic field data"""
        return self.data[7]

    def magTotalSqr(self):
        """Return the total magnetic field squared data |B|**2"""
        return self.data[5] ** 2 + self.data[6] ** 2 + self.data[7] ** 2

    def magnetic_pressure(self):
        """Return the magnetic pressure field data"""
        return self.magTotalSqr() * 0.5

    def pressure(self):
        """Return the thermal pressure field data"""
        thermal_en = (
            self.energy()
            - 0.5 * self.momTotalSqr() / self.dens()
            - self.magnetic_pressure()
        )
        pressure = (self.adi_idx - 1.0) * thermal_en
        return pressure

    def total_pressure(self):
        """Return the total pressure, magnetic plus thermal"""
        return self.pressure() + self.magnetic_pressure()

    def alfven_speed(self):
        """Return the Alfven speed for each mesh cell"""
        return np.sqrt(self.magTotalSqr() / self.dens())

    def fast_magnetosonic_speed_X(self):
        """Return the fast magnetosonic speed in the x direction for each mesh cell"""
        va2 = self.alfven_speed() ** 2
        vs2 = self.sound_speed() ** 2
        vax2 = self.magX() ** 2 / self.dens()
        quad = va2 + vs2 + np.sqrt((va2 + vs2) ** 2 - 4 * vax2 * vs2)
        return np.sqrt(0.5 * quad)

    def fast_magnetosonic_speed_Y(self):
        """Return the fast magnetosonic speed in the y direction for each mesh cell"""
        va2 = self.alfven_speed() ** 2
        vs2 = self.sound_speed() ** 2
        vay2 = self.magY() ** 2 / self.dens()
        quad = va2 + vs2 + np.sqrt((va2 + vs2) ** 2 - 4 * vay2 * vs2)
        return np.sqrt(0.5 * quad)

    def calculate_min_max_wave_speeds_X(self):
        """Return the the minimum and maximum wave speeds in the X direction
        
        Returns
        -------
        Tuple[ndarray, ndarray]
            the minimum and maximum wave speeds in the x direction for each cell
        """
        xvel = self.velX()
        cf = self.fast_magnetosonic_speed_X()
        lambda1 = xvel - cf
        lambda2 = xvel + cf

        return np.minimum(lambda1, lambda2), np.maximum(lambda1, lambda2)

    def calculate_min_max_wave_speeds_Y(self):
        """Return the the minimum and maximum wave speeds in the Y direction
        
        Returns
        -------
        Tuple[ndarray, ndarray]
            the minimum and maximum wave speeds in the y direction for each cell
        """
        yvel = self.velY()
        cf = self.fast_magnetosonic_speed_Y()
        lambda1 = yvel - cf
        lambda2 = yvel + cf

        return np.minimum(lambda1, lambda2), np.maximum(lambda1, lambda2)


class GravitySource:
    """Gravitational field sources
    
    This class provides gravity source terms for the equations. 
    
    Attributes
    ----------
    gravity_field : ndarray
        the gravitational field mesh data
    """

    def __init__(self, gravity_field):
        self.field = gravity_field

    def calculate_gravity_source(self, solvec):
        """Calculates the gravity field source terms
        
        Parameters
        ----------
        solvec : SolutionVector
            the solution vector of the simulation
        
        Returns
        -------
        gravity_source : ndarray
            the gravitational source term contribution to the system update
        """
        gravity_source = np.zeros(solvec.data.shape)
        gravity_source[1] = solvec.dens() * self.field[0]
        gravity_source[2] = solvec.dens() * self.field[1]
        gravity_source[3] = solvec.dens() * self.field[2]
        gravity_source[4] = (
            solvec.momX() * self.field[0]
            + solvec.momY() * self.field[1]
            + solvec.momZ() * self.field[2]
        )
        return gravity_source
