''' Input and Output utilities '''

import os
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import h5py

from matplotlib import rcParams

class Output:
    def __init__(self, Parameters, SolutionVector):
        self.dump_no = 0
        self.save_dir = 'output/'+str(Parameters.run_name)
        os.mkdir(self.save_dir)
        self.dump(SolutionVector)

    def dump(self, SolutionVector):
        file_name = self.save_dir+'/gawain_output_'+str(self.dump_no)+'.h5'
        self.dump_no+=1
        with h5py.File(file_name, 'w') as file:
            dataset = file.create_dataset('output_name', data=SolutionVector.data, dtype='f')



class Parameters:
    def __init__(self, input_file):
        self.input_module = import_module('tests.'+input_file)
        self.cfl = self.input_module.cfl
        self.mesh_shape = self.input_module.mesh_shape
        self.mesh_size = self.input_module.mesh_size
        self.t_max = self.input_module.t_max
        self.n_outputs = self.input_module.n_outputs
        self.with_gpu = self.input_module.with_gpu
        self.initial_condition = self.input_module.initial_condition
        self.boundary_conditions = self.input_module.boundary_conditions
        self.run_name = self.input_module.run_name
        self.cell_sizes = (self.mesh_size[0]/self.mesh_shape[0],
                           self.mesh_size[1]/self.mesh_shape[1],
                           self.mesh_size[2]/self.mesh_shape[2])

    def print_params(self):
        print('run name: ', self.run_name)
        print('clf condition =', self.cfl)
        print('nx, ny, nz =', self.mesh_shape)
        print('lx, ly, lz =', self.mesh_size)
        print('t-max=', self.t_max)
        print('-----------------------------------')


class Reader:
    def __init__(self, run_dir_path):
        self.file_path = run_dir_path
        self.data = []
        files = os.listdir(self.file_path)
        for num in range(len(files)):
            filename = self.file_path+'/gawain_output_'+str(num)+'.h5'
            file = h5py.File(filename, 'r')
            file_data = np.array(file['output_name'])
            self.data.append(file_data)
        self.data = np.array(self.data)

    def plot(self, variable, timesteps=[0], save_as=None):
        if len(self.data.shape)==2:
            fig, ax = plt.figure()
            ax.set_title('Plot of '+variable)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            ax.set_xlabel('x')
            ax.set_ylabel(variable)
            for step in timesteps:
                ax.plot(self.data[step], label='step='+str(step))
            ax.legend()
            if save_as:
                plt.savefig(save_as)
            plt.show()
        elif len(self.data.shape)==3:
            n_plots = len(timesteps)
            fig, axs = plt.subplots(1,n_plots, figsize=(5*n_plots, 5))
            fig.suptitle('Plots of '+variable)
            for step in timesteps:
                subplot = axs[timesteps.index(step)]
                subplot.pcolormesh(self.data[step], vmin=0, vmax=1, cmap='plasma')
                subplot.set_xlim(0, 100)
                subplot.set_ylim(0, 100)
                subplot.set_xlabel('x')
                subplot.set_ylabel('y')
                subplot.set_title('timestep='+str(step))
            if save_as:
                plt.savefig(save_as)
            plt.show()
        else:
            print('plot() only supports visualisation of 1D and 2D data')


    def animate(self, variable, save_as=None):
        if len(self.data.shape)==2:
            fig = plt.figure()
            plt.title('Animation of '+variable)
            plt.xlim(0, 100)
            plt.ylim(0, 1)
            plt.xlabel('x')
            plt.ylabel(variable)
            im = plt.plot(self.data[0], animated=True)

            def update_fig(i):
                im.set_array(self.data[i])
                return im,

            anim = FuncAnimation(fig, update_fig, blit=True)
            if save_as:
                anim.save(save_as)
            plt.show()

        elif len(self.data.shape)==3:
            fig = plt.figure()
            plt.title('Animation of '+variable)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.xlabel('x')
            plt.ylabel('y')
            im = plt.imshow(self.data[0], animated=True)

            def update_fig(i):
                im.set_array(self.data[i])
                return im,

            anim = FuncAnimation(fig, update_fig, blit=True)
            if save_as:
                anim.save(save_as)
            plt.show()

        else:
            print('plot() only supports visualisation of 1D and 2D data')


    def get_data(self, variable):
        return self.data
