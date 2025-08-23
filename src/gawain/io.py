"""Input and Output utilities

These classes provide the means of passing the initial
condtitions and options of the simulation to the code,
the output class for dumping the simulation result, and a
basic reading and plotting class to visualize the results.
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

import gawain.fluxes as fluxes
import gawain.integrators as integrator
import gawain.numerics as nu

if TYPE_CHECKING:
    from .numerics import MHDSolutionVector, SolutionVector


class Output:
    """HDF5 Output utilities class

    Tool for outputting data from the simulation to HDF5 format

    Attributes
    ----------
    save_dir : str
        the path to the save folder where all output data is saved
    hdf5_file : h5py.File
        the HDF5 file object for data storage
    """

    def __init__(
        self,
        Parameters: "Parameters",
        SolutionVector: "Union[SolutionVector, MHDSolutionVector]",
    ) -> None:
        """
        Parameters
        ----------
        Parameters : a Parameters object
            parameters and configuration of the simulation
        SolutionVector : a SolutionVector object
            the initial solution vector of the simulation
        """
        self.dump_no = 0
        self.save_file = str(Parameters.output_dir) + "/" + str(Parameters.run_name)
        self.hdf5_file = None
        # Create HDF5 file
        hdf5_filename = self.save_file + ".h5"
        if not os.path.exists(hdf5_filename):
            self.hdf5_file = h5py.File(hdf5_filename, "w")
        else:
            raise FileExistsError(
                f"File {hdf5_filename} already exists. Please choose a different run_name or delete the existing file."
            )
        parameters_to_save = {
            "variables": list,
            "adi_idx": float,
            "boundary_type": list,
            "cfl": float,
            "fluxer_type": str,
            "integrator_type": str,
            "mesh_shape": tuple,
            "mesh_size": tuple,
            "n_outputs": int,
            "run_name": str,
            "t_max": float,
            "with_mhd": bool,
        }
        # Store configuration and metadata
        config_group = self.hdf5_file.create_group("config")
        for key, dtype in parameters_to_save.items():
            value = getattr(Parameters, key)
            config_group.attrs[key] = dtype(value)

        self.hdf5_file.create_dataset("X", data=Parameters.mesh_grid[0])
        self.hdf5_file.create_dataset("Y", data=Parameters.mesh_grid[1])
        self.hdf5_file.create_dataset("Z", data=Parameters.mesh_grid[2])

        if Parameters.source_data is not None:
            self.hdf5_file.create_dataset(
                "source_function_field", data=Parameters.source_data
            )

        # Create datasets for time series data with unlimited time dimension
        data_shape = SolutionVector.data.shape
        maxshape = (None,) + data_shape
        self.solution_dataset = self.hdf5_file.create_dataset(
            "solutions",
            shape=(1,) + data_shape,
            maxshape=maxshape,
            dtype=SolutionVector.data.dtype,
        )

        self.timestamps_dataset = self.hdf5_file.create_dataset(
            "timestamps", shape=(1,), maxshape=(None,), dtype=float
        )

        # Store initial data
        self.solution_dataset[0] = SolutionVector.data
        self.timestamps_dataset[0] = 0.0
        self.hdf5_file.flush()

    def dump(
        self,
        SolutionVector: "Union[SolutionVector, MHDSolutionVector]",
        time: Optional[float] = None,
    ) -> None:
        """Append the solution to HDF5 file

        Parameters:
        -----------
        SolutionVector : a SolutionVector object
            the solution vector to be output to file
        time : float, optional
            the current simulation time
        """
        self.dump_no += 1

        # Resize datasets to accommodate new data
        self.solution_dataset.resize((self.dump_no + 1,) + SolutionVector.data.shape)
        self.timestamps_dataset.resize((self.dump_no + 1,))

        # Append new data
        self.solution_dataset[self.dump_no] = SolutionVector.data
        self.timestamps_dataset[self.dump_no] = (
            time if time is not None else self.dump_no
        )

        self.hdf5_file.flush()

    def close(self) -> None:
        """Close the HDF5 file"""
        if self.hdf5_file:
            self.hdf5_file.close()

    def __del__(self) -> None:
        """Destructor to ensure file is closed"""
        self.close()


class NPYOutput:
    """Output utilities class

    Tool for outputting data from the simulation

    Attributes
    ----------
    save_dir : str
        the path to the save folder where all output data is saved
    """

    def __init__(
        self,
        Parameters: "Parameters",
        SolutionVector: "Union[SolutionVector, MHDSolutionVector]",
    ) -> None:
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
        np.save(self.save_dir + "/X.npy", Parameters.mesh_grid[0])
        np.save(self.save_dir + "/Y.npy", Parameters.mesh_grid[1])
        np.save(self.save_dir + "/Z.npy", Parameters.mesh_grid[2])

        if Parameters.source_data is not None:
            np.save(
                self.save_dir + "/source_function_field.npy", Parameters.source_data
            )

    def dump(self, SolutionVector: "Union[SolutionVector, MHDSolutionVector]") -> None:
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

    def __init__(self, config: Dict[str, Any]) -> None:
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
        self.mesh_grid = config["mesh_grid"]
        self.t_max = config["t_max"]
        self.n_outputs = config["n_dumps"]
        self.adi_idx = config["adi_idx"]
        self.initial_condition = config[
            "initial_condition"
        ]  # .astype(np.float32)  # make this a f32 numpy array
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
        self.mesh_size = config["mesh_size"]
        if self.with_mhd:
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
        for i, axis in enumerate(self.boundary_type):
            if axis == "fixed":
                self.boundary_value[i] = [
                    self.initial_condition.take(0, axis=i + 1),
                    self.initial_condition.take(-1, axis=i + 1),
                ]
        config.pop("initial_condition", None)
        config.pop("source", None)
        config.pop("gravity", None)
        config.pop("mesh_grid", None)
        self.config = config

    def create_integrator(self) -> integrator.Integrator:
        """create an integrator object"""
        if self.integrator_type in self.available_integrators:
            return integrator.Integrator(self)
        else:
            raise KeyError(
                "Integrator not available, only the following type are available: {}".format(
                    self.available_integrators
                )
            )

    def create_fluxer(
        self,
    ) -> Union[
        fluxes.FluxCalculator,
        fluxes.HLLFluxer,
        fluxes.LaxWendroffFluxer,
        fluxes.LaxFriedrichsFluxer,
    ]:
        """create a flux calculation object"""
        if self.fluxer_type in self.available_fluxers:
            if self.fluxer_type == "base":
                return fluxes.FluxCalculator()
            elif self.fluxer_type == "hll":
                return fluxes.HLLFluxer()
            elif self.fluxer_type == "lax-wendroff":
                return fluxes.LaxWendroffFluxer()
            elif self.fluxer_type == "lax-friedrichs":
                return fluxes.LaxFriedrichsFluxer()
        else:
            raise KeyError(
                "Fluxer not available, only the following type are available: {}".format(
                    self.available_fluxers
                )
            )

    def create_source(self) -> Optional[np.ndarray]:
        """create a source object"""
        if self.source_data is not None:
            if self.source_data.shape == self.initial_condition.shape:
                return self.source_data
            else:
                raise TypeError(
                    "source data has inappropriate mesh shape, it must be the same shape as initial condition mesh shape"
                )
        return None

    def create_gravity(self) -> Optional[nu.GravitySource]:
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

    def print_params(self) -> None:
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
    """HDF5 Data loading and plotting utility

    Load simulation data from HDF5 file and plot results

    Attributes
    ----------
    file_path : str
        path to the HDF5 simulation file
    run_config : dict
        python dictionary containing run configuration
    data : dict
        dictionary containing raw simulation data for each variable
    """

    def __init__(self, hdf5_file_path: str) -> None:
        """
        Parameters
        ----------
        hdf5_file_path : str
            path to the HDF5 simulation file
        """
        self.file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

        # Load configuration from HDF5 attributes
        self.run_config = dict(self.hdf5_file["config"].attrs)
        self.data_dim = list(self.run_config["mesh_shape"]).count(1)

        # Load grid data
        self.grid = (
            np.squeeze(self.hdf5_file["X"][()]),
            np.squeeze(self.hdf5_file["Y"][()]),
            np.squeeze(self.hdf5_file["Z"][()]),
        )

        # Load times
        self.times = self.hdf5_file["timestamps"][()]

        # Load solution data and organize by variable
        solutions = self.hdf5_file["solutions"][()]
        self.data = {}
        for j, variable in enumerate(self.run_config["variables"]):
            self.data[variable] = solutions[:, j]

    def plot(
        self,
        variable: str,
        timesteps: List[int] = [0],
        save_as: Optional[str] = None,
        vmax: float = 1,
        vmin: float = 0,
    ) -> None:
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
            x = np.squeeze(self.grid[0])
            new_shape = tuple(filter(lambda x: x > 1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)
            fig, ax = plt.subplots()
            ax.set_title("Plot of " + variable)
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel("x")
            ax.set_ylabel(variable)
            for step in timesteps:
                ax.plot(
                    x, to_plot[step], label=f"step={step}, t={self.times[step]:.2f}"
                )
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
            if n_plots == 1:
                axs = [axs]
            fig.suptitle("Plots of " + variable)
            for step in timesteps:
                subplot = axs[timesteps.index(step)]
                subplot.pcolormesh(
                    self.grid[0],
                    self.grid[1],
                    to_plot[step],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="plasma",
                )
                subplot.set_xlabel("x")
                subplot.set_ylabel("y")
                subplot.set_title(f"step={step}, t={self.times[step]:.2f}")
            if save_as:
                plt.savefig(save_as)
            plt.show()
        else:
            print("plot() only supports visualisation of 1D and 2D data")

    def get_data(self, variable: str) -> np.ndarray:
        """Get the raw data for a particular variable"""
        return self.data[variable]

    def close(self) -> None:
        """Close the HDF5 file"""
        if self.hdf5_file:
            self.hdf5_file.close()

    def __del__(self) -> None:
        """Destructor to ensure file is closed"""
        self.close()
