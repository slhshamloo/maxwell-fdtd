import numpy as np
import copy

from matplotlib import pyplot as plt
from matplotlib import colors, patches

from .boundaries import AutoPML
from .objects import Slab


class Visualizer():
    def __init__(self, solver):
        self.solver = solver
        self.set_pos()
        self.set_color()
        self.set_cmap_norm()
    
    def set_pos(self, begin=(None, None, None), end=(None, None, None),
                crop_boundaries=True):
        self.crop_boundaries = crop_boundaries
        self.set_bounds(begin, end)
        self.set_cells()
    
    def set_color(self, color_E='blue', color_H='red', color_obj='lime',
                  cmap_E='Blues', cmap_H='Reds'):
        self.color_E, self.color_H, self.color_obj = \
            color_E, color_H, color_obj
        self.cmap_E, self.cmap_H = cmap_E, cmap_H
    
    def set_cmap_norm(self, norm_E='lin', norm_H='lin'):
        self.norm_E = get_norm_if_str(norm_E)
        self.norm_H = get_norm_if_str(norm_H)

    def plot1d(self, vertical=True, axis_space=0,
               slice_first_coordinate=0, slice_second_coordinate=0):
        fig, axs = plt.subplots(3, 2) if vertical else plt.subplots(2, 3)

        for i in range(3):
            ax_E = axs[i, 0] if vertical else axs[0, i]
            ax_H = axs[i, 1] if vertical else axs[1, i]

            self.plot1d_field(ax_E, 'E', axis_space, i,
                              slice_first_coordinate, slice_second_coordinate)
            self.plot1d_field(ax_H, 'H', axis_space, i,
                              slice_first_coordinate, slice_second_coordinate)
        
        plt.tight_layout()
        return fig, axs

    def plot2d(self, vertical=True, axis_slice=2, slice_coordinate=0):
        fig, axs = plt.subplots(3, 2) if vertical else plt.subplots(2, 3)

        for i in range(3):
            ax_E = axs[i, 0] if vertical else axs[0, i]
            ax_H = axs[i, 1] if vertical else axs[1, i]

            self.plot2d_field(ax_E, 'E', i, axis_slice, slice_coordinate)
            self.plot2d_field(ax_H, 'H', i, axis_slice, slice_coordinate)

        plt.tight_layout()
        return fig, axs

    def plot2d_magnitude(self, combine=False, vertical=True,
                         axis_slice=2, slice_coordinate=0):
        if combine:
            pass
        else:
            fig, axs = plt.subplots(2, 1) if vertical else plt.subplots(1, 2)
            self.plot2d_magnitude_field(axs[0], 'E', axis_slice,
                                        slice_coordinate)
            self.plot2d_magnitude_field(axs[1], 'H', axis_slice,
                                        slice_coordinate)
            plt.tight_layout()
            return fig, axs

    def plot2d_vector(self, combine=True, vertical=True,
                      axis_slice=2, slice_coordinate=0):
        pass

    def plot1d_field(self, ax, field_name, axis_space=0, axis_field=2,
                     slice_first_coordinate=0, slice_second_coordinate=0):
        field, color, cmap, norm = self.get_field_varables(field_name)

        ax.plot(*self.get_data_1d(field, axis_space, axis_field,
                                  slice_first_coordinate,
                                  slice_second_coordinate),
                c=color)

        if self.color_obj is not None:
            draw_object_1d(ax, *self.solver.objects,
                           axis_space=axis_space, color=self.color_obj)
        
        ax.set_xlabel(get_axis_name(axis_space))
        ax.set_ylabel(get_field_label(field_name, axis_field))
        ax.relim()

    def plot2d_field(self, ax, field_name, axis_field=2, axis_slice=2,
                     slice_coordinate=0):
        field, color, cmap, norm = self.get_field_varables(field_name)
        
        data_field = self.get_data_field_2d(
            field, axis_slice,slice_coordinate)[..., axis_field]
        pcm = ax.imshow(data_field.T, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=norm, cmap=cmap)
        plt.colorbar(pcm, ax=ax)

        if self.color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=self.color_obj)
        
        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(get_field_label(field_name, axis_field))

        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('auto')

    def plot2d_magnitude_field(self, ax, field_name, axis_slice=2,
                               slice_coordinate=0):
        field, color, cmap, norm = self.get_field_varables(field_name)
        
        data_field = np.sum(
            self.get_data_field_2d(field, axis_slice,slice_coordinate)**2,
            axis=2
        )**0.5
        pcm = ax.imshow(data_field.T, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=norm, cmap=cmap)
        plt.colorbar(pcm, ax=ax)

        if self.color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=self.color_obj)
        
        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(r'$\|\|\mathbf{' + field_name + r'}\|\|$')

        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('auto')
    
    def get_field_varables(self, field_name):
        if field_name == 'E':
            return self.solver.E, self.color_E, self.cmap_E, self.norm_E
        elif field_name == 'H':
            return self.solver.H, self.color_H, self.cmap_H, self.norm_H
    
    def get_data_1d(self, field, axis_space, axis_field,
                    slice_first_coordinate, slice_second_coordinate):
        begin_cell, end_cell = \
            self.begin_cell[axis_space], self.end_cell[axis_space]
        slice_first_cell, slice_second_cell = (
            int(slice_coordinate / self.solver.grid_dist) for slice_coordinate
            in (slice_first_coordinate, slice_second_coordinate)
        )
        
        data_space = np.linspace(
            self.begin_pos[axis_space], self.end_pos[axis_space],
            self.end_cell[axis_space] - self.begin_cell[axis_space]
        )

        slices = [slice_first_cell, slice_second_cell, axis_field]
        slices.insert(axis_space, slice(begin_cell, end_cell))
        data_field = field[tuple(slices)]

        return data_space, data_field
    
    def get_data_field_2d(self, field, axis_slice, slice_coordinate):
        slices = [slice(self.begin_cell[0], self.end_cell[0]),
                  slice(self.begin_cell[1], self.end_cell[1]),
                  slice(self.begin_cell[2], self.end_cell[2]),
                  slice(None)]
        slices.pop(axis_slice)
        slices.insert(axis_slice,
            int(round(slice_coordinate / self.solver.grid_dist))
        )
        return field[tuple(slices)]
    
    def get_data_space_2d(self, axis_slice):
        axis_space = list(range(3))
        axis_space.remove(axis_slice)
        return (
            np.linspace(self.begin_pos[axis_space[0]],
                        self.end_pos[axis_space[0]],
                        self.end_cell[axis_space[0]],
                        - self.begin_cell[axis_space[0]]),
            np.linspace(self.begin_pos[axis_space[1]],
                        self.end_pos[axis_space[1]],
                        self.end_cell[axis_space[1]]
                        - self.begin_cell[axis_space[1]])
        )
    
    def get_bounds_2d(self, axis_slice):
        axis_space = list(range(3))
        axis_space.remove(axis_slice)
        return (self.begin_pos[axis_space[0]], self.end_pos[axis_space[0]],
                self.begin_pos[axis_space[1]], self.end_pos[axis_space[1]])
    
    def set_bounds(self, begin, end):
        self.begin_pos, self.end_pos = list(begin), list(end)
        for i in range(3):
            self.set_axis_bounds(i)
        self.begin_pos, self.end_pos = \
            tuple(self.begin_pos), tuple(self.end_pos)
    
    def set_cells(self):
        self.begin_cell = tuple(int(round(begin_p / self.solver.grid_dist))
                                for begin_p in self.begin_pos)
        self.end_cell = tuple(int(round(end_p / self.solver.grid_dist))
                              for end_p in self.end_pos)
    
    def set_axis_bounds(self, axis_space):
        if self.begin_pos[axis_space] is None:
            self.begin_pos[axis_space] = 0
        if self.end_pos[axis_space] is None:
            self.end_pos[axis_space] = self.solver.length[axis_space]

        if self.crop_boundaries:
            for boundary in self.solver.boundaries:
                if isinstance(boundary, AutoPML):
                    begin_crop = boundary.begin_bound[axis_space]
                    end_crop = boundary.end_bound[axis_space]
                    if self.begin_pos[axis_space] < begin_crop:
                        self.begin_pos[axis_space] = begin_crop
                    if self.end_pos[axis_space] > end_crop:
                        self.end_pos[axis_space] = end_crop
                    break

def draw_object_1d(ax, *objects, axis_space=0, color='lime'):
    for obj in objects:
        ax.axvspan(obj.begin_pos[axis_space],
                   obj.end_pos[axis_space],
                   lw=0, alpha=0.5, color=color)


def draw_object_2d(ax, *objects, color='lime'):
    for obj in objects:
        if isinstance(obj, Slab):
            ax.add_patch(patches.Rectangle(
                (obj.begin_pos[0], obj.begin_pos[1]),
                obj.end_pos[0] - obj.begin_pos[0],
                obj.end_pos[1] - obj.begin_pos[1],
                lw=0, alpha=0.25, color=color
            ))


def get_norm_if_str(norm):
    if isinstance(norm, str):
        if norm == 'lin' or norm == 'linear':
            return colors.Normalize()
        elif norm == 'log' or norm == 'logarithmic':
            return colors.SymLogNorm(1e-5)
    else:
        return norm


def get_axis_name(axis_num):
    axes = ('x', 'y', 'z')
    return axes[axis_num]


def get_field_label(field_name, axis_field):
    return '$' + field_name + '_' + get_axis_name(axis_field) + '$'


def set_axis_labels_2d(ax, axis_slice):
    axis_space = list(range(3))
    axis_space.pop(axis_slice)
    ax.set_xlabel(get_axis_name(axis_space[0]))
    ax.set_ylabel(get_axis_name(axis_space[1]))
