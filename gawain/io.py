''' Input and Output utilities '''


from importlib import import_module


import h5py


def output_data(SolutionVector):
    pass



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

    def print_params(self):
        print('clf condition=', self.cfl)
        print('nx, ny, nz=', self.nx, self.ny, self.nz)
        print('t max=', self.t_max)
        print('-----------------------------------')
