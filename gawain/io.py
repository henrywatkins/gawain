''' Input and Output utilities '''


from importlib import import_module


import h5py

class Output:
    def __init__(self, Parameters, SolutionVector):
        self.dump_no = 0
        self.save_dir = '/output/'+str(Parameters.run_name)
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
        self.nx = self.input_module.nx
        self.ny = self.input_module.ny
        self.nz = self.input_module.nz
        self.t_max = self.input_module.t_max
        self.n_outputs = self.input_module.n_outputs
        self.with_gpu = self.input_module.with_gpu
        self.initial_condition = self.input_module.initial_condition
        self.boundary_conditions = self.input_module.boundary_conditions
        self.run_name = self.input_module.run_name

    def print_params(self):
        print('clf condition=', self.cfl)
        print('nx, ny, nz=', self.nx, self.ny, self.nz)
        print('t max=', self.t_max)
        print('-----------------------------------')


class Reader:
    def __init__(self, run_dir_path):
        self.file_path = run_dir_path

    def plot(self, variable, timestep=[0], slice=0 ,save_as=None):
        pass

    def animate(self, variable):
        pass

    def get_data(self, variable, timesteps=[0]):
        pass
