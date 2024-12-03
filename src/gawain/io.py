"""Input and Output utilities 

These classes provide the means of passing the initial 
condtitions and options of the simulation to the code, 
the output class for dumping the simulation result, and a
basic reading and plotting class to visualize the results.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

import gawain.integrators as integrator
import gawain.fluxes as fluxes
import gawain.numerics as nu


class Output:
    """Output utilities class

    Tool for outputting data from the simulation

    Attributes
    ----------
    save_dir : str
        the path to the save folder where all output data is saved
    """

    def __init__(self, Parameters, SolutionVector):
        """
        Parameters
        ----------
        Parameters : a Parameters object
            parameters and configuration of the simulation
        SolutionVector : a SolutionVector object
            the initial solution vector of the simulation
        """
        self.dump_no = 0
        self.save_dir = str(Parameters.output_dir) + "/" + str(Parameters.run_name)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.dump(SolutionVector)
        with open(self.save_dir + "/config.json", "w") as file:
            json.dump(Parameters.config, file)

        np.save(self.save_dir + "/initial_condition.npy", Parameters.initial_condition)

        if Parameters.source_data is not None:
            np.save(
                self.save_dir + "/source_function_field.npy", Parameters.source_data
            )

    def dump(self, SolutionVector):
        """Output the solution to file

        Parameters:
        -----------
        SolutionVector : a SolutionVector object
            the solution vector to be output to file
        """
        file_name = self.save_dir + "/gawain_output_" + str(self.dump_no) + ".npy"
        self.dump_no += 1
        np.save(file_name, SolutionVector.data)


class Parameters:
    """Configuration class for a simulation

    The Parameters objects holds all simulation configuration information.
    This includes mesh parameters, method, initial and boundary conditions.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            a python dictionary containing all configuration info
        """
        self.available_integrators = ["euler"]
        self.available_fluxers = ["base", "lax-wendroff", "lax-friedrichs", "hll"]
        self.integrator_type = config["integrator"]
        self.fluxer_type = config["fluxer"]
        self.run_name = config["run_name"]
        self.cfl = config["cfl"]
        self.mesh_shape = config["mesh_shape"]
        self.mesh_size = config["mesh_size"]
        self.t_max = config["t_max"]
        self.n_outputs = config["n_dumps"]
        self.adi_idx = config["adi_idx"]
        self.initial_condition = config["initial_condition"]
        self.source_data = config["source"] if "source" in config.keys() else None
        self.gravity_field = config["gravity"] if "gravity" in config.keys() else None
        self.boundary_type = config["boundary_type"]
        self.boundary_value = [[], [], []]
        self.output_dir = config["output_dir"]
        self.with_mhd = config["with_mhd"]
        self.cell_sizes = (
            self.mesh_size[0] / self.mesh_shape[0],
            self.mesh_size[1] / self.mesh_shape[1],
            self.mesh_size[2] / self.mesh_shape[2],
        )
        for i, axis in enumerate(self.boundary_type):
            if axis == "fixed":
                self.boundary_value[i] = [
                    self.initial_condition.take(0, axis=i + 1),
                    self.initial_condition.take(-1, axis=i + 1),
                ]
        config.pop("initial_condition", None)
        config.pop("source", None)
        config.pop("gravity", None)
        self.config = config

    def create_integrator(self):
        """create an integrator object"""
        if self.integrator_type in self.available_integrators:
            return integrator.Integrator(self)
        else:
            raise KeyError(
                "Integrator not available, only the following type are available: {}".format(
                    self.available_integrators
                )
            )

    def create_fluxer(self):
        """create a flux calculation object"""
        if self.fluxer_type in self.available_fluxers:
            if self.fluxer_type == "base":
                return fluxes.FluxCalculator()
            elif self.fluxer_type == "hll":
                return fluxes.HLLFluxer()
            elif self.fluxer == "lax-wendroff":
                return fluxes.LaxWendroffFluxer()
            elif self.fluxer == "lax-friedrichs":
                return fluxes.LaxFriedrichsFluxer()
        else:
            raise KeyError(
                "Fluxer not available, only the following type are available: {}".format(
                    self.available_fluxers
                )
            )

    def create_source(self):
        """create a source object"""
        if self.source_data is not None:
            if self.source_data.shape == self.initial_condition.shape:
                return self.source_data
            else:
                raise TypeError(
                    "source data has inappropriate mesh shape, it must be the same shape as initial condition mesh shape"
                )
        return None

    def create_gravity(self):
        """create a gravity source object"""
        if self.gravity_field is not None:
            if self.gravity_field.shape == (
                3,
                self.mesh_shape[0],
                self.mesh_shape[1],
                self.mesh_shape[2],
            ):
                return nu.GravitySource(self.gravity_field)
            else:
                raise TypeError(
                    "gravity field has inappropriate mesh shape, it must be (3, mesh_shape)"
                )
        return None

    def print_params(self):
        print("run name: ", self.run_name)
        print("clf condition =", self.cfl)
        print("nx, ny, nz =", self.mesh_shape)
        print("lx, ly, lz =", self.mesh_size)
        print("t max =", self.t_max)
        print("fluxer: ", str(self.fluxer_type))
        print("integrator: ", str(self.integrator_type))
        print("boundary types: ", self.boundary_type)
        print("-----------------------------------")


class Reader:
    """Data loading and plotting utility

    Load simulation data from a directory and plot results

    Attributes
    ----------
    file_path : str
        simulation run directory
    run_config : dict
        python dictionary containing run configuration
    data : dict
        dictionary containing raw simulation data for each variable
    """

    def __init__(self, run_dir_path):
        """
        Parameters
        ----------
        run_dir_path : str
            the simulation run directory
        """
        self.file_path = run_dir_path
        self.run_config = None
        with open(self.file_path + "/config.json", "r") as file:
            self.run_config = json.load(file)

        if self.run_config["with_mhd"]:
            self.variables = [
                "density",
                "xmomentum",
                "ymomentum",
                "zmomentum",
                "energy",
                "xmag",
                "ymag",
                "zmag",
            ]
        else:
            self.variables = [
                "density",
                "xmomentum",
                "ymomentum",
                "zmomentum",
                "energy",
            ]

        self.data_dim = self.run_config["mesh_shape"].count(1)

        self.data = {variable: [] for variable in self.variables}

        for i in range(self.run_config["n_dumps"]):
            filename = self.file_path + "/gawain_output_" + str(i) + ".npy"
            filedata = np.load(filename)
            for j, variable in enumerate(self.variables):
                self.data[variable].append(filedata[j])

        for variable in self.variables:
            self.data[variable] = np.stack(self.data[variable])

    def plot(self, variable, timesteps=[0], save_as=None, vmax=1, vmin=0):
        """Plot the output for a particular variable

        Parameters
        ----------
        variable : str
            the variable to plot
        timesteps : List[int], optional
            the indices of timesteps to plot
        save_as : str, optional
            the filename and path to save the plot image
        vmin : float, optional
            the minimum value to show on the plot
        vmax : float, optional
            the maximum value to show on the plot
        """
        to_plot = self.data[variable]
        # 1D runs
        if self.data_dim == 2:
            new_shape = tuple(filter(lambda x: x > 1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)
            fig, ax = plt.subplots()
            ax.set_title("Plot of " + variable)
            ax.set_xlim(0, new_shape[1])
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel("x")
            ax.set_ylabel(variable)
            for step in timesteps:
                ax.plot(to_plot[step], label="step=" + str(step))
            ax.legend()
            if save_as:
                plt.savefig(save_as)
            plt.show()

        # 2D runs
        elif self.data_dim == 1:

            new_shape = tuple(filter(lambda x: x > 1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)

            n_plots = len(timesteps)
            fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
            fig.suptitle("Plots of " + variable)
            for step in timesteps:
                subplot = axs[timesteps.index(step)]
                subplot.pcolormesh(
                    to_plot[step],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="plasma",
                )
                subplot.set_xlim(0, new_shape[2])
                subplot.set_ylim(0, new_shape[1])
                subplot.set_xlabel("x")
                subplot.set_ylabel("y")
                subplot.set_title("timestep=" + str(step))
            if save_as:
                plt.savefig(save_as)
            plt.show()
        else:
            print("plot() only supports visualisation of 1D and 2D data")

    def get_data(self, variable):
        """Get the raw data for a particular variable"""
        return self.data[variable]
