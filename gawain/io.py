""" Input and Output utilities """

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from matplotlib import rcParams
import h5py

import gawain.numerics as nu
import gawain.integrators as integrator
import gawain.fluxes as fluxes


class Output:
    def __init__(self, Parameters, SolutionVector):
        self.dump_no = 0
        self.save_dir = str(Parameters.output_dir) + "/" + str(Parameters.run_name)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.dump(SolutionVector)

    def dump(self, SolutionVector):
        file_name = self.save_dir + "/gawain_output_" + str(self.dump_no) + ".h5"
        self.dump_no += 1
        with h5py.File(file_name, "w") as file:
            for variable in SolutionVector.variable_names:
                to_output = SolutionVector.get_variable(variable)
                dataset = file.create_dataset(variable, data=to_output, dtype="f")


class Parameters:
    def __init__(self, **kwargs):
        self.integrator_type = None
        self.fluxer_type = None
        self.run_name = kwargs["run_name"]
        self.cfl = kwargs["cfl"]
        self.mesh_shape = kwargs["mesh_shape"]
        self.mesh_size = kwargs["mesh_size"]
        self.t_max = kwargs["t_max"]
        self.n_outputs = kwargs["n_outputs"]
        self.adi_idx = kwargs["adi_idx"]
        self.initial_condition = kwargs["initial_condition"]
        self.boundary_type = kwargs["boundary_type"]
        self.boundary_value = [[], [], []]
        self.output_dir = kwargs["output_dir"]
        self.with_mhd = kwargs["with_mhd"]
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

        if kwargs["integrator"] == "euler":
            self.integrator_type = integrator.Integrator
        elif kwargs["integrator"] == "rk2":
            self.integrator_type = integrator.RK2Integrator
        elif kwargs["integrator"] == "leapfrog":
            self.integrator_type = integrator.LeapFrogIntegrator
        elif kwargs["integrator"] == "predictor-corrector":
            self.integrator_type = integrator.PredictorCorrectorIntegrator
        else:
            print("integrator type not recognised, defalting to euler integrator")
            self.integrator_type = integrator.Integrator

        if kwargs["fluxer"] == "base":
            self.fluxer_type = fluxes.FluxCalculator
        elif kwargs["fluxer"] == "lax-wendroff":
            self.fluxer_type = fluxes.LaxWendroffFluxer
        elif kwargs["fluxer"] == "lax-friedrichs":
            self.fluxer_type = fluxes.LaxFriedrichsFluxer
        elif kwargs["fluxer"] == "vanleer":
            self.fluxer_type = fluxes.VanLeerFluxer
        elif kwargs["fluxer"] == "hll":
            self.fluxer_type = fluxes.HLLFluxer
        else:
            print("Fluxer not recognized, defaulting to default flux")
            self.fluxer_type = fluxes.FluxCalculator

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
        self.data = {}
        files = os.listdir(self.file_path)
        for num in range(len(files)):
            filename = self.file_path + "/gawain_output_" + str(num) + ".h5"
            file = h5py.File(filename, "r")
            for variable in file.keys():
                file_data = np.array(file[variable])
                if variable in self.data.keys():
                    self.data[variable].append(file_data)
                else:
                    self.data[variable] = [file_data]

        for variable in self.data.keys():
            self.data[variable] = np.array(self.data[variable])
        self.data_dim = self.data["density"].shape.count(1)

        self.variables = self.data.keys()

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
