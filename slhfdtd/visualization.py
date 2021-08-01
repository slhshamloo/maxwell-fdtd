import numpy as np
import copy

from matplotlib import pyplot as plt
from matplotlib import colors, patches

from .boundaries import AutoPML
from .objects import Slab


class Visualizer():
    def __init__(self, solver):
        self.solver = solver

    def plot1d_E(self, ax, axis_space=0, axis_E=2,
                 slice_first_coordinate=0, slice_second_coordinate=0,
                 begin_space=None, end_space=None, crop_boundaries=True,
                 color='blue', obj_color='lightskyblue'):
        self.plot1d_field(ax, self.solver.E, axis_space, axis_E,
                          slice_first_coordinate, slice_second_coordinate,
                          begin_space, end_space, crop_boundaries,
                          color, obj_color)

    def plot1d_H(self, ax, axis_space=0, axis_H=1,
                 slice_first_coordinate=0, slice_second_coordinate=0,
                 begin_space=None, end_space=None, crop_boundaries=True,
                 color='red', obj_color='lightskyblue'):
        self.plot1d_field(ax, self.solver.H, axis_space, axis_H,
                          slice_first_coordinate, slice_second_coordinate,
                          begin_space, end_space, crop_boundaries,
                          color, obj_color)

    def plot2d_E(self, ax, slice_z=0,
                 begin_x=None, begin_y=None, end_x=None, end_y=None,
                 crop_boundaries=True, norm='lin', cmap='Blues',
                 obj_color='lime'):
        self.plot2d_field(ax, self.solver.E, slice_z,
                          begin_x, begin_y, end_x, end_y,
                          crop_boundaries, norm, cmap, obj_color)

    def plot2d_H(self, ax, slice_z=0,
                 begin_x=None, begin_y=None, end_x=None, end_y=None,
                 crop_boundaries=True, norm='lin', cmap='Reds',
                 obj_color='lime'):
        self.plot2d_field(ax, self.solver.H, slice_z,
                          begin_x, begin_y, end_x, end_y,
                          crop_boundaries, norm, cmap, obj_color)
    
    def draw_object_1d(self, ax, *objects, axis_space=0, color='lime'):
        for obj in objects:
            ax.axvspan(obj.begin_pos[axis_space],
                       obj.end_pos[axis_space],
                       lw=0, alpha=0.5, color=color)
    
    def draw_object_2d(self, ax, *objects, color='lime'):
        for obj in objects:
            if isinstance(obj, Slab):
                ax.add_patch(patches.Rectangle(
                    (obj.begin_pos[0], obj.begin_pos[1]),
                    obj.end_pos[0] - obj.begin_pos[0],
                    obj.end_pos[1] - obj.begin_pos[1],
                    lw=0, alpha=0.25, color=color
                ))
    
    def plot1d_field(self, ax, field, axis_space=0, axis_field=2,
                     slice_first_coordinate=0, slice_second_coordinate=0,
                     begin_space=None, end_space=None, crop_boundaries=True,
                     color='blue', obj_color='lightskyblue'):
        if begin_space is None:
            begin_space = 0
        if end_space is None:
            end_space = self.solver.length[axis_space]

        if crop_boundaries:
            for boundary in self.solver.boundaries:
                if isinstance(boundary, AutoPML):
                    begin_crop = boundary.begin_bound[axis_space]
                    end_crop = boundary.end_bound[axis_space]
                    if begin_space < begin_crop:
                        begin_space = begin_crop
                    if end_space > end_crop:
                        end_space = end_crop
                    break

        begin_cell = round(begin_space / self.solver.grid_dist)
        end_cell = round(end_space / self.solver.grid_dist)
        slice_first_cell = round(slice_first_coordinate
                                 / self.solver.grid_dist)
        slice_second_cell = round(slice_second_coordinate
                                  / self.solver.grid_dist)

        if axis_space == 0:
            data_field = field[begin_cell:end_cell, slice_first_cell,
                               slice_second_cell, axis_field]
        if axis_space == 1:
            data_field = field[slice_first_cell, begin_cell:end_cell,
                               slice_second_cell, axis_field]
        if axis_space == 2:
            data_field = field[slice_first_cell, slice_second_cell,
                               begin_cell:end_cell, axis_field]

        data_space = np.linspace(begin_space, end_space,
                                 end_cell - begin_cell)
        ax.plot(data_space, data_field, c=color)

        if obj_color is not None:
            self.draw_object_1d(ax, *self.solver.objects,
                                axis_space=axis_space,
                                color=obj_color)
        
        ax.relim()
    
    def plot2d_field(self, ax, field, slice_z=0,
                     begin_x=None, begin_y=None, end_x=None, end_y=None,
                     crop_boundaries=True, norm='lin', cmap='jet',
                     obj_color='lime'):
        if begin_x is None:
            begin_x = 0
        if end_x is None:
            end_x = self.solver.length[0]
        if begin_y is None:
            begin_y = 0
        if end_y is None:
            end_y = self.solver.length[1]

        if crop_boundaries:
            for boundary in self.solver.boundaries:
                if isinstance(boundary, AutoPML):
                    begin_crop_x = boundary.begin_bound[0]
                    end_crop_x = boundary.end_bound[0]
                    begin_crop_y = boundary.begin_bound[1]
                    end_crop_y = boundary.end_bound[1]
                    if begin_x < begin_crop_x:
                        begin_x = begin_crop_x
                    if begin_y < begin_crop_y:
                        begin_y = begin_crop_y
                    if end_x > end_crop_x:
                        end_x = end_crop_x
                    if end_y > end_crop_y:
                        end_y = end_crop_y
                    break

        begin_x_cell = round(begin_x / self.solver.grid_dist)
        begin_y_cell = round(begin_y / self.solver.grid_dist)
        end_x_cell = round(end_x / self.solver.grid_dist)
        end_y_cell = round(end_y / self.solver.grid_dist)
        slice_z_cell = round(slice_z / self.solver.grid_dist)

        data_field = np.sum(field[begin_x_cell:end_x_cell,
                                  begin_y_cell:end_y_cell,
                                  slice_z_cell, :]**2,
                            axis=2)**0.5

        if isinstance(norm, str):
            if norm == 'lin':
                norm = colors.Normalize()
            elif norm == 'log':
                norm = colors.SymLogNorm(1e-5)

        pcm = ax.imshow(data_field.T, origin='lower', norm=norm,
                        extent=(begin_x, end_x, begin_y, end_y),
                        cmap=cmap)
        plt.colorbar(pcm, ax=ax)

        if obj_color is not None:
            self.draw_object_2d(ax, *self.solver.objects,
                                color=obj_color)

        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('auto')
