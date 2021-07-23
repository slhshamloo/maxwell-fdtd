import numpy as np
from matplotlib import pyplot as plt


class Visualizer():
    def __init__(self, solver):
        self.solver = solver
    
    def plot1d_field(self, artist, field, axis_space=0, axis_field=2,
                 slice_first_coordinate=0, slice_second_coordinate=0,
                 begin_space=0, end_space=None):
        axis_length = self.solver.length[axis_space]
        
        if end_space is None:
            end_space = axis_length
        
        begin_cell = round(begin_space / self.solver.grid_dist)
        end_cell = round(end_space / self.solver.grid_dist)
        slice_first_cell = round(slice_first_coordinate / self.solver.grid_dist)
        slice_second_cell = round(slice_first_coordinate / self.solver.grid_dist)
        
        if axis_space == 0:
            data_field = field[begin_cell:end_cell, slice_first_cell,
                slice_second_cell, axis_field]
        if axis_space == 1:
            data_field = field[slice_first_cell, begin_cell:end_cell,
                slice_second_cell, axis_field]
        if axis_space == 2:
            data_field = field[slice_first_cell, slice_second_cell,
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
    
    def plot2d_field(self, ax, field, color_map, slice_z=0,
                     begin_x=0, begin_y=0, end_x=None, end_y=None):
        if end_x is None:
            end_x = self.solver.length[0]
        if end_y is None:
            end_y = self.solver.length[0]
        
        begin_x_cell = round(begin_x / self.solver.grid_dist)
        begin_y_cell = round(begin_y / self.solver.grid_dist)
        end_x_cell = round(end_x / self.solver.grid_dist)
        end_y_cell = round(end_y / self.solver.grid_dist)
        slice_z_cell = round(slice_z / self.solver.grid_dist)
        
        data_field = np.sum(field[begin_x_cell:end_x_cell,
                               begin_y_cell:end_y_cell,
                               slice_z_cell, :]**2,
                            axis=2)**0.5
        ax.imshow(data_field, cmap=color_map)
    
    def plot2d_E(self, ax, color_map='Blues', slice_z=0, begin_x=0, begin_y=0,
                 end_x=None, end_y=None):
        self.plot2d_field(ax, self.solver.E, color_map, slice_z,
                          begin_x, begin_y, end_x, end_y)
    
    def plot2d_H(self, ax, color_map='Reds', slice_z=0, begin_x=0, begin_y=0,
                 end_x=None, end_y=None):
        self.plot2d_field(ax, self.solver.H, color_map, slice_z,
                          begin_x, begin_y, end_x, end_y)
