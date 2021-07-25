import numpy as np


class Object:
    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity


class Slab(Object):
    def __init__(self, begin_x, begin_y, begin_z, end_x, end_y, end_z,
                 permittivity=1.0, permeability=1.0, conductivity=0.0):
        super().__init__(permittivity, permeability, conductivity)
        self.begin_pos = (begin_x, begin_y, begin_z)
        self.end_pos = (end_x, end_y, end_z)
    
    def set_solver(self, solver):
        self.set_pos(solver.grid_dist)
        self.set_solver_parameters(solver)
    
    def set_pos(self, grid_dist):
        self.begin_cell = tuple(round(begin / grid_dist)
                                for begin in self.begin_pos)
        self.end_cell = tuple(round(end / grid_dist)
                              for end in self.end_pos)
        self.shape = (*(end_c - begin_c + 1
            for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            3)
        self.slices = (*(slice(begin_c, end_c + 1)
            for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            slice(None))
    
    def set_solver_parameters(self, solver):
        self.set_solver_permittivity(solver)
        self.set_solver_permeability(solver)
        self.set_solver_conductivity(solver)
        
    def set_solver_permittivity(self, solver):
        solver.inverse_permittivity[self.slices] = np.ones(
            self.shape) / self.permittivity
    
    def set_solver_permeability(self, solver):
        solver.inverse_permeability[self.slices] = np.ones(
            self.shape) / self.permeability
    
    def set_solver_conductivity(self, solver):
        dissipation = np.ones(self.shape
            ) * 0.5* self.conductivity / self.permittivity
        solver.dissipation_mult[self.slices] = (1 - dissipation) / (1 + dissipation)
        solver.dissipation_add[self.slices] = 1 / (1 + dissipation)
        