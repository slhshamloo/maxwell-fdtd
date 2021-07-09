import numpy as np
from matplotlib import pyplot as plt


class Visualizer():
    def __init__(self, solver):
        self.solver = solver
    
    def plot1d_field(self, artist, field, axis_space=0, axis_field=2,
                 slice_first_coordinate=0, slice_second_coordinate=0,
                 begin_space=0, end_space=None):
        axis_length = (self.solver.length_x, self.solver.length_y,
                       self.solver.length_z)[axis_space]
        axis_cell_count = (self.solver.cell_count_x, self.solver.cell_count_y,
                           self.solver.cell_count_z)[axis_space]
        
        if end_space is None:
            end_space = axis_length
        
        begin_cell = round(begin_space / axis_length * axis_cell_count)
        end_cell = round(end_space / axis_length * axis_cell_count)
        
        if axis_space == 0:
            data_field = field[begin_cell:end_cell, slice_first_coordinate,
                slice_second_coordinate, axis_field]
        if axis_space == 1:
            data_field = field[slice_first_coordinate, begin_cell:end_cell,
                slice_second_coordinate, axis_field]
        if axis_space == 2:
            data_field = field[slice_first_coordinate, slice_second_coordinate,
                begin_cell:end_cell, axis_field]
        
        data_space = np.linspace(begin_space, end_space, end_cell - begin_cell)
        artist.set_data(data_space, data_field)
    
    def plot1d_E(self, artist, axis_space=0, axis_E=2,
                 slice_first_coordinate=0, slice_second_coordinate=0,
                 begin_space=0, end_space=None):
        self.plot1d_field(artist, self.solver.E, axis_space, axis_E,
                 slice_first_coordinate, slice_second_coordinate,
                 begin_space, end_space)
    
    def plot1d_H(self, artist, axis_space=0, axis_H=1,
                 slice_first_coordinate=0, slice_second_coordinate=0,
                 begin_space=0, end_space=None):
        self.plot1d_field(artist, self.solver.H, axis_space, axis_H,
                 slice_first_coordinate, slice_second_coordinate,
                 begin_space, end_space)
