from math import pi, sin
import numpy as np


class Source:
    def __init__(self, begin_x, begin_y, begin_z, end_x, end_y, end_z,
                 power, freq=600e12, phase=0.0):
        self.begin_x, self.begin_y, self.begin_z = begin_x, begin_y, begin_z
        self.end_x, self.end_y, self.end_z = end_x, end_y, end_z
        self.power, self.omega, self.phase = power, 2*pi * freq, phase 
        
        self.current_time_step=0
    
    def set_solver(self, solver):
        self.solver = solver
        self.set_pos(solver.grid_dist)
        self.set_amplitude()
    
    def set_pos(self, grid_dist):
        self.begin_cell_x = round(self.begin_x / grid_dist)
        self.begin_cell_y = round(self.begin_y / grid_dist)
        self.begin_cell_z = round(self.begin_z / grid_dist)
        self.end_cell_x = round(self.end_x / grid_dist)
        self.end_cell_y = round(self.end_y / grid_dist)
        self.end_cell_z = round(self.end_z / grid_dist)
    
    def set_amplitude(self):
        self.amplitude = (self.power * self.solver.inverse_permittivity[
            self.begin_cell_x:self.end_cell_x + 1,
            self.begin_cell_y:self.end_cell_y + 1,
            self.begin_cell_z:self.end_cell_z + 1,
            2])**0.5
    
    def step(self):
        self.update_E()
        self.update_H()
        self.current_time_step += 1
    
    def update_E(self):
        self.solver.E[self.begin_cell_x:self.end_cell_x + 1,
                      self.begin_cell_y:self.end_cell_y + 1,
                      self.begin_cell_z:self.end_cell_z + 1,
                      2] += self.amplitude * sin(
                          self.omega * self.current_time_step
                          * self.solver.time_step + self.phase)
    
    def update_H(self):
        pass

class PointSource(Source):
    def __init__(self, x, y, z, power, freq=600e12, phase=0.0):
        super().__init__(x, y, z, x, y, z, power, freq, phase)


class LineSource(Source):
    def set_solver(self, solver):
        self.solver = solver
        self.set_pos(solver.grid_dist)
        self.set_span()
        self.set_amplitude()
    
    def set_span(self):
        length = int(((self.end_cell_x - self.begin_cell_x)**2 +
                      (self.end_cell_y - self.begin_cell_y)**2 +
                      (self.end_cell_z - self.begin_cell_z)**2)**0.5)
        self.x = np.linspace(self.begin_cell_x, self.end_cell_x, length)
        self.y = np.linspace(self.begin_cell_y, self.end_cell_y, length)
        self.z = np.linspace(self.begin_cell_z, self.end_cell_z, length)
    
    def set_amplitude(self):
        self.amplitude = (self.power * self.solver.inverse_permittivity[
            self.x.astype(np.int), self.y.astype(np.int),
            self.z.astype(np.int), 2])**0.5
        
    def update_E(self):
        self.solver.E[self.x.astype(np.int),
            self.y.astype(np.int), self.z.astype(np.int), 2] += (
                self.amplitude * sin(
                self.omega * self.current_time_step
                * self.solver.time_step + self.phase))
