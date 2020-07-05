''' Input and Output utilities '''

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
        self.save_dir = str(Parameters.output_dir)+'/'+str(Parameters.run_name)
        if not os.path.exists(self.save_dir):
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
            to_output = SolutionVector.energy()
            dataset = file.create_dataset('energy', data=to_output, dtype='f')
            to_output = SolutionVector.momTotalSqr()
            dataset = file.create_dataset('sqrmomentum', data=to_output, dtype='f')



class Parameters:
    def __init__(self, input_dict):
        self.integrator_type = None
        self.fluxer_type = None
        self.run_name = None
        self.cfl = None
        self.mesh_shape = None
        self.mesh_size = None
        self.t_max = None
        self.n_outputs = None
        self.adi_idx = None
        self.with_gpu = False
        self.initial_condition = None
        self.boundary_type = None
        self.boundary_value = [None,None,None]
        self.cell_sizes = None
        self.output_dir = '.'
        self.with_mhd = False
        
        self.set_parameters(input_dict)

    def set_parameters(self, dict_input):
        self.cfl = dict_input['cfl']
        self.mesh_shape = dict_input['mesh_shape']
        self.mesh_size = dict_input['mesh_size']
        self.t_max = dict_input['t_max']
        self.n_outputs = dict_input['n_dumps']
        self.using_gpu = dict_input['using_gpu']
        self.initial_condition = dict_input['initial_con']
        self.boundary_type = dict_input['bound_cons']
        self.run_name = dict_input['run_name']
        self.with_mhd = dict_input['with_mhd']
        self.output_dir = dict_input['output_dir']
        self.adi_idx = dict_input['adi_idx']
        self.cell_sizes = (self.mesh_size[0]/self.mesh_shape[0],
                           self.mesh_size[1]/self.mesh_shape[1],
                           self.mesh_size[2]/self.mesh_shape[2])
        for i, axis in enumerate(self.boundary_type):
            if axis=="fixed":
                self.boundary_value[i] = [self.initial_condition.take(0, axis=i+1),
                                          self.initial_condition.take(-1, axis=i+1)]
                                          
        if dict_input['integrator']=='euler':
            self.integrator_type = integrator.Integrator
        elif dict_input['integrator']=='rk2':
            self.integrator_type = integrator.RK2Integrator
        elif dict_input['integrator']=='leapfrog':
            self.integrator_type = integrator.LeapFrogIntegrator
        elif dict_input['integrator']=='predictor-corrector':
            self.integrator_type = integrator.PredictorCorrectorIntegrator
        else:
            print("integrator type not recognised, defalting to euler integrator")
            self.integrator_type = integrator.Integrator
            
        
        if dict_input['fluxer']=='base':
            self.fluxer_type = fluxes.FluxCalculator
        elif dict_input['fluxer']=='lax-wendroff':
            self.fluxer_type = fluxes.LaxWendroffFluxer
        elif dict_input['fluxer']=='lax-friedrichs':
            self.fluxer_type = fluxes.LaxFriedrichsFluxer
        elif dict_input['fluxer']=='vanleer':
            self.fluxer_type = fluxes.VanLeerFluxer
        elif dict_input['fluxer']=='hll':
            self.fluxer_type = fluxes.HLLFluxer
        else:
            print("Fluxer not recognized, defaulting to default flux")
            self.fluxer_type = fluxes.FluxCalculator


    def print_params(self):
        print('run name: ', self.run_name)
        print('clf condition =', self.cfl)
        print('nx, ny, nz =', self.mesh_shape)
        print('lx, ly, lz =', self.mesh_size)
        print('t max =', self.t_max)
        print('fluxer: ', str(self.fluxer_type))
        print('integrator: ', str(self.integrator_type))
        print('boundary types: ',self.boundary_type)
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
        # 1D runs
        if self.data_dim==2:
            new_shape = tuple(filter(lambda x: x>1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)
            fig, ax = plt.subplots()
            ax.set_title('Plot of '+variable)
            ax.set_xlim(0, new_shape[1])
            ax.set_ylim(0, to_plot[0].max())
            ax.set_xlabel('x')
            ax.set_ylabel(variable)
            for step in timesteps:
                ax.plot(to_plot[step], label='step='+str(step))
            ax.legend()
            if save_as:
                plt.savefig(save_as)
            plt.show()
        
        #2D runs
        elif self.data_dim==1:

            new_shape = tuple(filter(lambda x: x>1, to_plot.shape))
            to_plot = to_plot.reshape(new_shape)

            n_plots = len(timesteps)
            fig, axs = plt.subplots(1,n_plots, figsize=(5*n_plots, 5))
            fig.suptitle('Plots of '+variable)
            for step in timesteps:
                subplot = axs[timesteps.index(step)]
                subplot.pcolormesh(to_plot[step], vmin=0, vmax=to_plot[0].max(), cmap='plasma')
                subplot.set_xlim(0, new_shape[2])
                subplot.set_ylim(0, new_shape[1])
                subplot.set_xlabel('x')
                subplot.set_ylabel('y')
                subplot.set_title('timestep='+str(step))
            if save_as:
                plt.savefig(save_as)
            plt.show()
        else:
            print('plot() only supports visualisation of 1D and 2D data')


    def get_data(self, variable):
        return self.data[variable]
