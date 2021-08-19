from slhfdtd.solving import Solver
import numpy as np


class Object:
    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        self.permittivity, self.permeability, self.conductivity = \
            permittivity, permeability, conductivity
        self.begin_pos, self.end_pos = None, None
    
    def set_solver(self, solver):
        self.solver = solver
        self.set_pos()
        self.set_solver_parameters()
    
    def set_pos(self):
        self.begin_pos = list(b if b > 0 else 0 for b in self.begin_pos)
        self.end_pos = list(e if e < se else se - self.solver.grid_dist
            for (e, se) in zip(self.end_pos, self.solver.length)
        )
        for i in range(3):
            if self.end_pos[i] < self.begin_pos[i]:
                self.end_pos[i] = self.begin_pos[i]
        
        self.begin_cell = list(int(round(b / self.solver.grid_dist))
                               for b in self.begin_pos)
        self.end_cell = list(int(round(e / self.solver.grid_dist))
                             for e in self.end_pos)
        for i in range(3):
            if self.end_cell[i] == self.begin_cell[i]:
                self.end_cell[i] += 1
        
        self.frame = (
            *(slice(begin_c, end_c)
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            slice(None)
        )

        self.mask = None
    
    def set_solver_parameters(self):
        self.set_solver_permittivity()
        self.set_solver_permeability()
        self.set_solver_conductivity()

    def set_solver_permittivity(self):
        if type(self.permittivity) is np.ndarray:
            self.permittivity = self.permittivity[:, :, :, None]
        self.solver.permittivity[self.frame] += \
            self.mask * (self.permittivity - 1)

    def set_solver_permeability(self):
        if type(self.permeability) is np.ndarray:
            self.permeability = self.permeability[:, :, :, None]
        self.solver.permeability[self.frame] += \
            self.mask * (self.permeability - 1)

    def set_solver_conductivity(self):
        if type(self.conductivity) is np.ndarray:
            self.conductivity = self.conductivity[:, :, :, None]
        dissipation = self.mask * 0.5 * self.conductivity / self.permittivity
        self.solver.dissipation_mult[self.frame] = \
            (1 - dissipation) / (1 + dissipation)
        self.solver.dissipation_add[self.frame] = 1 / (1 + dissipation)


class Slab(Object):
    def __init__(self, begin, end, permittivity=1.0, permeability=1.0,
                 conductivity=0.0):
        super().__init__(permittivity, permeability, conductivity)
        self.begin_pos, self.end_pos = begin, end

    def set_pos(self):
        super().set_pos()
        shape = (
            *(end_c - begin_c
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            3
        )
        self.mask = np.ones(shape)


class Ball(Object):
    def __init__(self, center, outer_radius, inner_radius=0.0,
                 permittivity=1.0, permeability=1.0, conductivity=0.0):
        super().__init__(permittivity, permeability, conductivity)
        self.center, self.outer_radius, self.inner_radius = \
            center, outer_radius, inner_radius

        self.begin_pos = tuple(c - outer_radius for c in center)
        self.end_pos = tuple(c + outer_radius for c in center)

    def set_solver(self, solver):
        self.solver = solver
        self.set_pos()
        self.set_solver_parameters()
    
    def set_pos(self):
        slices = tuple(slice(b, e, self.solver.grid_dist)
                       for (b, e) in zip(self.begin_pos, self.end_pos))
        grid = np.mgrid[slices]
        dist = ((g - c) ** 2 for (g, c) in zip(grid, self.center))
        self.mask = ((dist < self.outer_radius + self.solver.grid_dist)
                     & (dist > self.inner_radius - self.solver.grid_dist))
