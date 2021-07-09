from math import pi, sin


class Source:
    def __init__(self, begin_x, begin_y, begin_z, end_x, end_y, end_z,
                 power, freq=600e12, phase=0.0):
        self.begin_x, self.begin_y, self.begin_z = begin_x, begin_y, begin_z
        self.end_x, self.end_y, self.end_z = end_x, end_y, end_z
        self.power = power
        self.omega = 2*pi * freq
        
        self.current_time_step=0
    
    def set_solver(self, solver):
        self.set_pos(solver.length_x, solver.length_y, solver.length_z,
            solver.cell_count_x, solver.cell_count_y, solver.cell_count_z)
        self.set_amplitude()
    
    def set_pos(self, length_x, length_y, length_z,
                cell_count_x, cell_count_y, cell_count_z):
        self.begin_cell_x = round(self.begin_x / length_x * cell_count_x)
        self.begin_cell_y = round(self.begin_y / length_y * cell_count_y)
        self.begin_cell_z = round(self.begin_z / length_z * cell_count_z)
        self.end_cell_x = round(self.end_x / length_x * cell_count_x)
        self.end_cell_y = round(self.end_y / length_y * cell_count_y)
        self.end_cell_z = round(self.end_z / length_z * cell_count_z)
    
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

class PointSource(Source):
    def __init__(self, x, y, z, power, freq=600e12, phase=0.0):
        super.__init__(x, y, z, x, y, z, power, freq, phase)
