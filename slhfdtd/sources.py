from math import pi, sin, floor
import numpy as np


class Source:
    def __init__(self, begin_x, begin_y, begin_z, end_x, end_y, end_z,
                 power, freq=600e12, phase=0.0, func=sin, direction=2):
        self.begin_pos = (begin_x, begin_y, begin_z)
        self.end_pos = (end_x, end_y, end_z)
        self.power, self.omega, self.phase = power, 2*pi * freq, phase
        self.func = func
        self.direction = direction
        
        self.current_time_step=0
    
    def set_solver(self, solver):
        self.solver = solver
        self.set_pos(solver.grid_dist)
        self.set_amplitude()
    
    def set_pos(self, grid_dist):
        self.begin_cell = list(round(begin / grid_dist)
                               for begin in self.begin_pos)
        self.end_cell = list(round(end / grid_dist)
                             for end in self.end_pos)
        for i in range(3):
            if self.end_cell[i] == self.begin_cell[i]:
                self.end_cell[i] += 1
        
        self.shape = (*(end_c - begin_c
            for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            3)
        self.slices = (*(slice(begin_c, end_c)
            for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            int(self.direction))
    
    def set_amplitude(self):
        self.amplitude = (self.power * self.solver.inverse_permittivity[
            self.slices])**0.5
    
    def step(self):
        self.update_E()
        self.update_H()
        self.current_time_step += 1
    
    def update_E(self):
        self.solver.E[self.slices] += self.amplitude * self.func(
            self.omega * self.current_time_step
            * self.solver.time_step + self.phase)
    
    def update_H(self):
        pass

class PointSource(Source):
    def __init__(self, x, y, z, power, freq=600e12, phase=0.0, func=sin):
        super().__init__(x, y, z, x, y, z, power, freq, phase, func)


class LineSource(Source):
    def set_solver(self, solver):
        self.solver = solver
        self.set_pos(solver.grid_dist)
        self.set_span()
        self.set_amplitude()
    
    def set_span(self):
        length = int(sum((end_c - begin_c)**2
            for (begin_c, end_c) in zip(self.begin_cell, self.end_cell))**0.5)
        
        self.pos = list()
        for (begin_c, end_c) in zip(self.begin_cell, self.end_cell):
            if begin_c == end_c - 1:
                self.pos.append(np.ones((length,)).astype(np.int)
                                * int(begin_c))
            else:
                self.pos.append(
                    np.linspace(begin_c, end_c, length).astype(np.int))
    
    def set_amplitude(self):
        self.amplitude = (self.power
            * self.solver.inverse_permittivity[
                (*self.pos, 2)])**0.5
        
    def update_E(self):
        self.solver.E[self.pos[0], self.pos[1], self.pos[2], 2] += (
            self.amplitude * self.func(
            self.omega * self.current_time_step
            * self.solver.time_step + self.phase))


def pulse(theta):
    if floor(theta/pi) % 2 == 0:
        return 1
    else:
        return -1
