import numpy as np


class Object:
    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        self.permittivity, self.permeability, self.conductivity = \
            permittivity, permeability, conductivity


class Slab(Object):
    def __init__(self, begin, end, permittivity=1.0, permeability=1.0,
                 conductivity=0.0):
        super().__init__(permittivity, permeability, conductivity)
        self.begin_pos, self.end_pos = begin, end

    def set_solver(self, solver):
        self.solver = solver
        self.set_pos()
        self.set_solver_parameters()

    def set_pos(self):
        self.begin_cell = list(round(begin / self.solver.grid_dist)
                               for begin in self.begin_pos)
        self.end_cell = list(round(end / self.solver.grid_dist)
                             for end in self.end_pos)
        for i in range(3):
            if self.end_cell[i] == self.begin_cell[i]:
                self.end_cell[i] += 1

        self.shape = (
            *(end_c - begin_c
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            3
        )
        self.pos = (
            *(slice(begin_c, end_c)
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            slice(None)
        )

    def set_solver_parameters(self):
        self.set_solver_permittivity()
        self.set_solver_permeability()
        self.set_solver_conductivity()

    def set_solver_permittivity(self):
        self.solver.inverse_permittivity[self.pos] = (
            np.ones(self.shape) / self.permittivity
        )

    def set_solver_permeability(self):
        self.solver.inverse_permeability[self.pos] = (
            np.ones(self.shape) / self.permeability
        )

    def set_solver_conductivity(self):
        dissipation = (
            np.ones(self.shape)
            * 0.5 * self.conductivity / self.permittivity
        )
        self.solver.dissipation_mult[self.pos] = \
            (1 - dissipation) / (1 + dissipation)
        self.solver.dissipation_add[self.pos] = 1 / (1 + dissipation)
