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


class Output:
    def __init__(self, Parameters, SolutionVector):
        self.dump_no = 0
        self.save_dir = str(Parameters.output_dir) + "/" + str(Parameters.run_name)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.dump(SolutionVector)
        with open(self.save_dir + "/config.json", "w") as file:
            json.dump(Parameters.config, file)

    def dump(self, SolutionVector):
        file_name = self.save_dir + "/gawain_output_" + str(self.dump_no) + ".npy"
        self.dump_no += 1
        np.save(file_name, SolutionVector.data)


class Parameters:
    def __init__(self, config):
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
        self.config = config

    def create_integrator(self):
        if self.integrator_type in self.available_integrators:
            return integrator.Integrator(self)
        else:
            raise KeyError(
                "Integrator not available, only the following type are available: {}".format(
                    self.available_integrators
                )
            )

    def create_fluxer(self):
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
    def __init__(self, run_dir_path):
        self.file_path = run_dir_path
        self.run_config = None
        files = os.listdir(self.file_path)
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

    def plot(self, variable, timesteps=[0], save_as=None):
        to_plot = self.data[variable]
        # 1D runs
        if self.data_dim == 2:
            new_shape = tuple(filter(lambda x: x > 1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)
            fig, ax = plt.subplots()
            ax.set_title("Plot of " + variable)
            ax.set_xlim(0, new_shape[1])
            ax.set_ylim(0, to_plot[0].max())
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
                    to_plot[step], vmin=0, vmax=to_plot[0].max(), cmap="plasma"
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
        return self.data[variable]
