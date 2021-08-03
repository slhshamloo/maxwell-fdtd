import numpy as np
import copy

from matplotlib import pyplot as plt
from matplotlib import colors, patches, gridspec

from .boundaries import AutoPML
from .objects import Slab


class Visualizer():
    def __init__(self, solver):
        self.solver = solver
        self.set_fields()
        self.set_pos()
        self.set_color()
        self.set_cmap_norm()
        self.set_orientation()
    
    def set_fields(self, fields = ('E', 'H', 'S')):
        self.fields = fields
    
    def set_pos(self, begin=(None, None, None), end=(None, None, None),
                crop_boundaries=True):
        self.crop_boundaries = crop_boundaries
        self.set_bounds(begin, end)
        self.set_cells()
    
    def set_color(self, color_E='blue', color_H='red', color_S='purple',
                  color_obj='lime', cmap_energy='Purples',
                  cmap_E='Blues', cmap_H='Reds', cmap_S='jet'):
        self.color_E, self.color_H, self.color_S, self.color_obj = \
            color_E, color_H, color_S, color_obj
        self.cmap_E, self.cmap_H, self.cmap_S, self.cmap_energy = \
            cmap_E, cmap_H, cmap_S, cmap_energy
    
    def set_cmap_norm(self, norm_E='lin', norm_H='lin', norm_S='lin',
                      norm_energy='lin'):
        self.norm_E, self.norm_H, self.norm_S, self.norm_energy = (
            get_norm_if_str(norm) for norm
            in (norm_E, norm_H, norm_S, norm_energy)
        )
    
    def set_orientation(self, orientation='v'):
        if orientation.lower() in ('v', 'vertical', 'vert', 'ver'):
            self.orientation = 'v'
        elif orientation.lower() in ('h', 'horizontal', 'horiz', 'hor'):
            self.orientation = 'h'
        elif orientation.lower() in ('c', 'centered', 'center', 'cen'):
            self.orientation = 'c'
    
    def get_field_varables(self, field_name):
        if field_name.upper() == 'E':
            return self.solver.E, self.color_E, self.cmap_E, self.norm_E
        elif field_name.upper() == 'H':
            return self.solver.H, self.color_H, self.cmap_H, self.norm_H
        elif field_name.upper() == 'S':
            return (self.solver.get_poynting(), self.color_S,
                    self.cmap_S, self.norm_S)

    def plot1d(self, axis_space=0,
               slice_first_coordinate=0, slice_second_coordinate=0):
        if self.orientation == 'v':
            fig, axs = plt.subplots(3, len(self.fields))
        elif self.orientation == 'h':
            fig, axs = plt.subplots(len(self.fields), 3)

        for i in range(3):
            for j in range(len(self.fields)):
                ax = axs[i, j] if self.orientation == 'v' else axs[j, i]
                self.plot1d_field(ax, self.fields[j], axis_space, i,
                                  slice_first_coordinate,
                                  slice_second_coordinate)

        plt.tight_layout()
        return fig, axs

    def plot2d(self, axis_slice=2, slice_coordinate=0):
        if self.orientation == 'v':
            fig, axs = plt.subplots(3, len(self.fields))
        elif self.orientation == 'h':
            fig, axs = plt.subplots(len(self.fields), 3)

        for i in range(3):
            for j in range(len(self.fields)):
                ax = axs[i, j] if self.orientation == 'v' else axs[j, i]
                self.plot2d_field(ax, self.fields[j], i, axis_slice,
                                  slice_coordinate)

        plt.tight_layout()
        return fig, axs

    def plot2d_magnitude(self, axis_slice=2, slice_coordinate=0):
        if self.orientation == 'v':
            fig, axs = plt.subplots(3, len(self.fields))
        elif self.orientation == 'h':
            fig, axs = plt.subplots(len(self.fields), 3)

        for i in len(self.fields):
            self.plot2d_magnitude_field(axs[i], self.fields[i],
                                        axis_slice, slice_coordinate)
        plt.tight_layout()
        return fig, axs

    def plot2d_vector(self, combine=True,
                      axis_slice=2, slice_coordinate=0):
        if combine:
            pass
        else:
            if self.orientation == 'v':
                fig, axs = plt.subplots(3, len(self.fields))
            elif self.orientation == 'h':
                fig, axs = plt.subplots(len(self.fields), 3)
            elif self.orientation == 'c':
                if len(self.fields) == 3:
                    fig = plt.figure()
                    gs = gridspec.GridSpec(4, 4)
                    axs = np.array((plt.subplot(gs[:2, :2]),
                                    plt.subplot(gs[:2, 2:]),
                                    plt.subplot(gs[2:4, 1:3])))

            for i in len(self.fields):
                self.plot2d_vector_field(axs[i], self.fields[i],
                                            axis_slice, slice_coordinate)
            plt.tight_layout()
            return fig, axs

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
    
    def plot2d_vector_field(self, ax, field_name, axis_slice=2,
                            slice_coordinate=0):
        field, color, cmap, norm = self.get_field_varables(field_name)
        
        data_field, data_space = (
            self.get_data_field_2d(field, axis_slice,slice_coordinate),
            self.get_data_space_2d(axis_slice)
        )

        

        if self.color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=self.color_obj)
        
        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(r'$\mathbf{' + field_name + r'}$')

        ax.relim()
    
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
        if norm.lower() in ('lin', 'linear'):
            return colors.Normalize()
        elif norm.lower() in ('log', 'logarithmic'):
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
