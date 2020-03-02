''' Input and Output utilities '''

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from matplotlib import rcParams
import h5py

class Output:
    def __init__(self, Parameters, SolutionVector):
        self.dump_no = 0
        self.save_dir = 'output/'+str(Parameters.run_name)
        if os.path.exists(self.save_dir):
            Parameters.run_name += "_new"
            new_name = Parameters.run_name
            print('A run folder with that name already exists, changing directory name to ', new_name)
            self.save_dir = 'output/'+str(Parameters.run_name)
            os.mkdir(self.save_dir)
        else:
            os.mkdir(self.save_dir)
        self.dump(SolutionVector)

    def dump(self, SolutionVector):
        file_name = self.save_dir+'/gawain_output_'+str(self.dump_no)+'.h5'
        self.dump_no+=1
        with h5py.File(file_name, 'w') as file:
            to_output = SolutionVector.dens()
            dataset = file.create_dataset('density', data=to_output, dtype='f')
            to_output = SolutionVector.momX()
            dataset = file.create_dataset('xmomentum', data=to_output, dtype='f')
            to_output = SolutionVector.momY()
            dataset = file.create_dataset('ymomentum', data=to_output, dtype='f')
            to_output = SolutionVector.en()
            dataset = file.create_dataset('energy', data=to_output, dtype='f')
            to_output = SolutionVector.momMagSqr()
            dataset = file.create_dataset('sqrmomentum', data=to_output, dtype='f')



class Parameters:
    def __init__(self, from_file=None):
        self.run_name = None
        self.cfl = None
        self.mesh_shape = None
        self.mesh_size = None
        self.t_max = None
        self.n_outputs = None
        self.adi_idx = None
        self.with_gpu = False
        self.initial_condition = None
        self.boundary_conditions = None
        self.cell_sizes = None

        if from_file is not None:
            with open(from_file, 'rb') as input:
                input_dict = pickle.load(input)
            self.set_parameters(input_dict)
        else:
            print('Invalid or missing input file, please specify parameters')

    def set_parameters(self, dict_input):
        self.cfl = dict_input['clf']
        self.mesh_shape = dict_input['mesh_shape']
        self.mesh_size = dict_input['mesh_size']
        self.t_max = dict_input['t_max']
        self.n_outputs = dict_input['n_dumps']
        self.with_gpu = dict_input['using_gpu']
        self.initial_condition = dict_input['initial_con']
        self.boundary_conditions = dict_input['bound_cons']
        self.run_name = dict_input['run_name']
        self.adi_idx = dict_input['adi_idx']
        self.cell_sizes = (self.mesh_size[0]/self.mesh_shape[0],
                           self.mesh_size[1]/self.mesh_shape[1],
                           self.mesh_size[2]/self.mesh_shape[2])


    def print_params(self):
        print('run name: ', self.run_name)
        print('clf condition =', self.cfl)
        print('nx, ny, nz =', self.mesh_shape)
        print('lx, ly, lz =', self.mesh_size)
        print('t max =', self.t_max)
        print('-----------------------------------')


class Reader:
    def __init__(self, run_dir_path):
        self.file_path = run_dir_path
        self.variables = ['density','xmomentum','ymomentum','sqrmomentum', 'energy']
        self.data = {variable:[] for variable in self.variables}
        files = os.listdir(self.file_path)
        for num in range(len(files)):
            filename = self.file_path+'/gawain_output_'+str(num)+'.h5'
            file = h5py.File(filename, 'r')
            for variable in self.variables:
                file_data = np.array(file[variable])
                self.data[variable].append(file_data)
        for variable in self.variables:
            self.data[variable] = np.array(self.data[variable])
        self.data_dim = self.data['density'].shape.count(1)

    def plot(self, variable, timesteps=[0], save_as=None):
        to_plot = self.data[variable]
        if self.data_dim==2:
            new_shape = tuple(filter(lambda x: x>1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)
            fig, ax = plt.subplots()
            ax.set_title('Plot of '+variable)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            ax.set_xlabel('x')
            ax.set_ylabel(variable)
            for step in timesteps:
                ax.plot(to_plot[step], label='step='+str(step))
            ax.legend()
            if save_as:
                plt.savefig(save_as)
            plt.show()
        elif self.data_dim==1:

            new_shape = tuple(filter(lambda x: x>1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)

            n_plots = len(timesteps)
            fig, axs = plt.subplots(1,n_plots, figsize=(5*n_plots, 5))
            fig.suptitle('Plots of '+variable)
            for step in timesteps:
                subplot = axs[timesteps.index(step)]
                subplot.pcolormesh(to_plot[step], vmin=0, vmax=1, cmap='plasma')
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
        to_plot = self.data[variable]
        if self.data_dim==2:
            new_shape = tuple(filter(lambda x: x>1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)
            fig = plt.figure()
            plt.title('Animation of '+variable)
            plt.xlim(0, 100)
            plt.ylim(0, 1)
            plt.xlabel('x')
            plt.ylabel(variable)
            im = plt.plot(to_plot[0], animated=True)

            def update_fig(i):
                im.set_array(to_plot[i])
                return im,

            anim = FuncAnimation(fig, update_fig, blit=True)
            if save_as:
                anim.save(save_as)
            plt.show()

        elif self.data_dim==1:

            new_shape = tuple(filter(lambda x: x>1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)

            fig = plt.figure()
            plt.title('Animation of '+variable)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.xlabel('x')
            plt.ylabel('y')
            im = plt.imshow(to_plot[0], animated=True)

            def update_fig(i):
                im.set_array(to_plot[i])
                return im,

            anim = FuncAnimation(fig, update_fig, blit=True)
            if save_as:
                anim.save(save_as)
            plt.show()

        else:
            print('plot() only supports visualisation of 1D and 2D data')


    def get_data(self, variable):
        return self.data
