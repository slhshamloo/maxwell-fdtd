import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors, patches, gridspec, cm

from copy import copy
from math import log10

from .boundaries import AutoPML
from .objects import Slab


class Visualizer():
    def __init__(self, solver):
        self.solver = solver
        self.set_pos()
        self.set_fields()
        self.set_colors()
        self.set_norms()
        self.set_interpolation_2d()
        self.set_orientation()
        self.set_figsize()
        self.set_aspect()
    
    def set_pos(self, begin=(None, None, None), end=(None, None, None),
                crop_boundaries=True):
        self.crop_boundaries = crop_boundaries
        self.set_bounds(begin, end)
        self.set_cells()
    
    def set_fields(self, fields = ('E', 'H', 'S', 'U')):
        self.fields = fields
    
    def set_colors(self, field_colors=('blue', 'red', 'purple', 'green'),
                   cmaps=('Blues', 'Reds', 'Purples', 'jet'),
                   object_colors=('lime', 'lime', 'lime', 'white')):
        self.field_colors, self.cmaps, self.object_colors \
            = field_colors, cmaps, object_colors

    def set_norms(self, norms='lin'):
        self.norms = norms
    
    def set_interpolation_2d(self, interpolation=None):
        self.interpolation = interpolation
    
    def set_orientation(self, orientation='c'):
        if orientation.lower() in ('v', 'vertical', 'vert', 'ver'):
            self.orientation = 'v'
        elif orientation.lower() in ('h', 'horizontal', 'horiz', 'hor'):
            self.orientation = 'h'
        elif orientation.lower() in ('c', 'centered', 'center', 'cen'):
            self.orientation = 'c'
    
    def set_figsize(self, figsize=(7, 5)):
        self.figsize = figsize
    
    def set_aspect(self, aspect='auto'):
        self.aspect = aspect
    
    def set_color_field(self, field_name, field_color=None, cmap=None,
                        object_color='lime'):
        field_index = self.fields.index(field_name)

        if field_color is not None:
            if isinstance(self.field_colors, str):
                self.field_colors = len(self.fields) * [self.field_colors]
                self.field_colors[field_index] = field_color
            else:
                self.field_colors = (self.field_colors[:field_index]
                                     + field_color
                                     + self.field_colors[field_index + 1:])

        if cmap is not None:
            if isinstance(self.cmaps, str):
                self.cmaps = len(self.fields) * [self.cmaps]
                self.cmaps[field_index] = cmap
            else:
                self.cmaps = (self.cmaps[:field_index] + cmap
                              + self.cmaps[field_index + 1:])

        if self.object_colors is None or isinstance(self.object_colors, str):
            self.object_colors = len(self.fields) * [self.object_colors]
            self.object_colors[field_index] = object_color
        else:
            self.object_colors = (self.object_colors[:field_index]
                                  + object_color
                                  + self.object_colors[field_index + 1:])
    
    def set_cmap_norm_field(self, field_name, norm):
        field_index = self.fields.index(field_name)

        if isinstance(self.norms, str):
            self.norms = len(self.fields) * [self.norms]
            self.norms[field_index] = norm
        else:
            self.norms = (self.norms[:field_index] + norm
                          + self.norms[field_index + 1:])
    
    def add_field(self, field_name, field_index=None,
                  field_color='black', cmap='jet',
                  object_color='white', norm='lin'):
        if field_index is None:
            field_index = len(self.fields)
        self.fields = list(self.fields)
        self.fields.insert(field_index, field_name)

        if not isinstance(self.field_colors, str):
            self.field_colors = list(self.field_colors)
            self.field_colors.insert(field_index, field_color)
        if not isinstance(self.cmaps, str):
            self.cmaps = list(self.cmaps)
            self.cmaps.insert(field_index, cmap)
        if self.object_colors is not None and \
                not isinstance(self.object_colors, str):
            self.object_colors = list(self.object_colors)
            self.object_colors.insert(field_index, object_color)
        if not isinstance(self.norms, str):
            self.norms = list(self.norms)
            self.norms.insert(field_index, norm)
    
    def delete_field(self, field_name):
        field_index = self.fields.index(field_name)
        self.fields = list(self.fields)
        self.fields.pop(field_index)

        if not isinstance(self.field_colors, str):
            self.field_colors = list(self.field_colors)
            self.field_colors.pop(field_index)
        if not isinstance(self.cmaps, str):
            self.cmaps = list(self.cmaps)
            self.cmaps.pop(field_index)
        if self.object_colors is not None and \
                not isinstance(self.object_colors, str):
            self.object_colors = list(self.object_colors)
            self.object_colors.pop(field_index)
        if not isinstance(self.norms, str):
            self.norms = list(self.norms)
            self.norms.pop(field_index)

    def plot1d(self, axis_space=0,
               slice_first_coordinate=0, slice_second_coordinate=0):
        fields = list(self.fields)
        if 'U' in (field.upper() for field in fields):
            fields.remove('U')
        fig, axs = self.get_fig_and_axs(3, len(fields))

        for i in range(3):
            for j in range(len(fields)):
                ax = axs[i, j] if self.orientation == 'v' else axs[j, i]
                self.plot1d_field(ax, fields[j], axis_space, i,
                                  slice_first_coordinate,
                                  slice_second_coordinate)

        plt.tight_layout()
        return fig, axs

    def plot2d(self, axis_slice=2, slice_coordinate=0):
        fields = list(self.fields)
        if 'U' in (field.upper() for field in fields):
            fields.remove('U')
        fig, axs = self.get_fig_and_axs(3, len(fields))

        for i in range(3):
            for j in range(len(fields)):
                ax = axs[i, j] if self.orientation == 'v' else axs[j, i]
                self.plot2d_field(ax, fields[j], i, axis_slice,
                                  slice_coordinate)

        plt.tight_layout()
        return fig, axs

    def plot2d_magnitude(self, axis_slice=2, slice_coordinate=0):
        fig, axs = self.get_fig_and_axs(1, len(self.fields))

        for i in range(len(self.fields)):
            if self.fields[i].upper() == 'U':
                self.plot2d_energy(axs[i], axis_slice, slice_coordinate)
            else:
                self.plot2d_magnitude_field(axs[i], self.fields[i],
                                            axis_slice, slice_coordinate)

        plt.tight_layout()
        return fig, axs

    def plot2d_vector(self, axis_slice=2, slice_coordinate=0,
                      combine=True, resolution=25,
                      quiver=True, stream=False):
        fields = list(self.fields)
        if 'U' in (field.upper() for field in fields):
            fields.remove('U')

        if combine:
            fig, axs = plt.subplots()

            for i in range(len(fields)):
                self.plot2d_vector_field(axs, fields[i],
                                         axis_slice, slice_coordinate,
                                         resolution, quiver, stream)

            axs.set_title(', '.join(r'$\mathbf{' + field + r'}$'
                                    for field in fields))
        else:
            fig, axs = self.get_fig_and_axs(1, len(fields))

            for i in range(len(fields)):
                self.plot2d_vector_field(axs[i], fields[i],
                                         axis_slice, slice_coordinate,
                                         resolution, quiver, stream)

        plt.tight_layout()
        return fig, axs
    
    def plot2d_vect_on_mag(self, axis_slice=2, slice_coordinate=0,
                           resolution=25, arrow_color='black',
                           quiver=True, stream=False):
        fields = list(self.fields)
        if 'U' in (field.upper() for field in fields):
            fields.remove('U')
        fig, axs = self.get_fig_and_axs(1, len(fields))

        for i in range(len(fields)):
            self.plot2d_vect_on_mag_field(axs[i], fields[i],
                axis_slice, slice_coordinate, resolution, arrow_color,
                quiver, stream
            )

        plt.tight_layout()
        return fig, axs

    def plot1d_field(self, ax, field_name, axis_space=0, axis_field=2,
                     slice_first_coordinate=0, slice_second_coordinate=0):
        field, color, color_obj, _, _ = \
            self.get_field_varables(field_name)

        ax.plot(*self.get_data_1d(field, axis_space, axis_field,
                                  slice_first_coordinate,
                                  slice_second_coordinate),
                c=color)

        if color_obj is not None:
            draw_object_1d(ax, *self.solver.objects,
                           axis_space=axis_space, color=color_obj)
        
        ax.set_xlabel('$' + get_axis_name(axis_space) + '$')
        ax.set_ylabel(get_field_label(field_name, axis_field))
        ax.set_aspect(self.aspect)

    def plot2d_field(self, ax, field_name, axis_field=2, axis_slice=2,
                     slice_coordinate=0):
        field, _, color_obj, cmap, norm = \
            self.get_field_varables(field_name)
        
        data_field = self.get_data_field_2d(
            field, axis_slice,slice_coordinate)[..., axis_field]
        norm = get_norm_if_str(norm, data_field.min(), data_field.max())

        pcm = ax.imshow(data_field.T, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=norm, cmap=cmap,
                        interpolation=self.interpolation)
        plt.colorbar(pcm, ax=ax)

        if color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=color_obj)
        
        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(get_field_label(field_name, axis_field))
        ax.set_aspect(self.aspect)

    def plot2d_magnitude_field(self, ax, field_name, axis_slice=2,
                               slice_coordinate=0):
        field, color, color_obj, cmap, norm = \
            self.get_field_varables(field_name)
        
        data_field = np.sum(
            self.get_data_field_2d(field, axis_slice,slice_coordinate)**2,
            axis=2
        )**0.5
        norm = get_norm_if_str(norm, data_field.min(), data_field.max())

        pcm = ax.imshow(data_field.T, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=norm, cmap=cmap,
                        interpolation=self.interpolation)
        plt.colorbar(pcm, ax=ax)

        if color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=color_obj)
        
        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(r'$\|\,\mathbf{' + field_name + r'}\,\|$')
        ax.set_aspect(self.aspect)
    
    def plot2d_energy(self, ax, axis_slice=2, slice_coordinate=0):
        field, _, color_obj, cmap, norm = self.get_field_varables('U')

        data_field = self.get_data_field_2d(field, axis_slice,
                                            slice_coordinate)
        norm = get_norm_if_str(norm, field.min(), field.max())
        pcm = ax.imshow(data_field.T, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=norm, cmap=cmap,
                        interpolation=self.interpolation)
        plt.colorbar(pcm, ax=ax)

        if color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=color_obj)
        
        set_axis_labels_2d(ax, axis_slice)
        ax.set_title('$U$')
        ax.set_aspect(self.aspect)
    
    def plot2d_vector_field(self, ax, field_name, axis_slice=2,
                            slice_coordinate=0, resolution=25,
                            quiver=True, stream=False):
        field, _, color_obj, cmap, norm = \
            self.get_field_varables(field_name)
        
        data_space, data_field = (self.get_data_space_2d(axis_slice),
            self.get_data_field_2d(field, axis_slice, slice_coordinate)
        )

        data_field = np.delete(data_field, axis_slice, -1)
        data_space, data_field = self.reduce_data(data_space, data_field,
                                                  resolution=resolution)
        field_magnitude = np.sum(data_field**2, axis=2)**0.5
        norm = get_norm_if_str(norm, field_magnitude.min(),
                               field_magnitude.max())

        if stream and (data_field >= 1e-9).any():
            ax.streamplot(*data_space, data_field[..., 0], data_field[..., 1],
                          color=field_magnitude, cmap=cmap, norm=norm,
                          density=resolution/30)
        if quiver:
            data_field[..., 0] /= field_magnitude
            data_field[..., 1] /= field_magnitude
            ax.quiver(*data_space, data_field[..., 0], data_field[..., 1],
                      field_magnitude, cmap=cmap, norm=norm,
                      scale=resolution, headwidth=5)

        bounds = self.get_bounds_2d(axis_slice)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, ax=ax)
        cb.ax.set_xlabel(r'\|\,\mathbf{' + field_name + r'}\,\|$')

        if color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=color_obj)

        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(r'$\mathbf{' + field_name + r'}$')
        ax.set_aspect(self.aspect)
    
    def plot2d_vect_on_mag_field(self, ax, field_name,
                                 axis_slice=2, slice_coordinate=0,
                                 resolution=25, arrow_color='black',
                                 quiver=True, stream=False):
        field, _, color_obj, cmap, norm = \
            self.get_field_varables(field_name)
        
        data_space, data_field = (self.get_data_space_2d(axis_slice),
            self.get_data_field_2d(field, axis_slice, slice_coordinate)
        )

        field_magnitude = np.sum(data_field**2, axis=2)**0.5
        norm = get_norm_if_str(norm, field_magnitude.min(),
                               field_magnitude.max())
        pcm = ax.imshow(field_magnitude, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=norm, cmap=cmap,
                        interpolation=self.interpolation)
        cb = plt.colorbar(pcm, ax=ax)
        cb.ax.set_xlabel(r'$\|\,\mathbf{' + field_name + r'}\,\|$')

        data_field = np.delete(data_field, axis_slice, -1)
        data_space, data_field = self.reduce_data(data_space, data_field,
                                                  resolution=resolution)
        field_magnitude = np.sum(data_field**2, axis=2)**0.5

        if stream and (data_field >= 1e-9).any():
            ax.streamplot(*data_space, data_field[..., 0], data_field[..., 1],
                          color=arrow_color, density=resolution/30)
        if quiver:
            data_field[..., 0] /= field_magnitude
            data_field[..., 1] /= field_magnitude
            ax.quiver(*data_space, data_field[..., 0], data_field[..., 1],
                      color=arrow_color, scale=resolution, headwidth=5)

        if color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=color_obj)

        set_axis_labels_2d(ax, axis_slice)
        ax.set_title(r'$\mathbf{' + field_name + r'}$')
        ax.set_aspect(self.aspect) 

    def plot2d_poynting_on_energy(self, ax, axis_slice=2, slice_coordinate=0,
                                  resolution=25, poynting_cmap='Greys',
                                  quiver=True, stream=False):
        energy, color, color_obj, energy_cmap, energy_norm = \
            self.get_field_varables('U')
        poynting, _, _, _, poynting_norm = self.get_field_varables('S')
        poynting_cmap = get_cmap_if_str(poynting_cmap)
        
        data_space, data_energy, data_poynting = (
            self.get_data_space_2d(axis_slice),
            self.get_data_field_2d(energy, axis_slice, slice_coordinate),
            self.get_data_field_2d(poynting, axis_slice, slice_coordinate)
        )
        energy_norm = get_norm_if_str(energy_norm, data_energy.min(),
                                      data_energy.max())
        pcm = ax.imshow(data_energy, extent=self.get_bounds_2d(axis_slice),
                        origin='lower', norm=energy_norm, cmap=energy_cmap,
                        interpolation=self.interpolation)
        cb = plt.colorbar(pcm, ax=ax)
        cb.ax.set_xlabel('$U$')

        data_poynting = np.delete(data_poynting, axis_slice, -1)
        data_space, data_poynting = self.reduce_data(data_space, data_poynting,
                                                     resolution=resolution)
        poynting_magnitude = np.sum(data_poynting**2, axis=2)**0.5
        poynting_norm = get_norm_if_str(poynting_norm, data_poynting.min(),
                                        data_poynting.max())

        if stream and (data_poynting >= 1e-9).any():
            ax.streamplot(*data_space,
                          data_poynting[..., 0], data_poynting[..., 1],
                          color=poynting_magnitude, cmap=poynting_cmap,
                          norm=poynting_norm, density=resolution/30)
        if quiver:
            data_poynting[..., 0] /= poynting_magnitude
            data_poynting[..., 1] /= poynting_magnitude
            ax.quiver(*data_space,
                      data_poynting[..., 0], data_poynting[..., 1],
                      poynting_magnitude, cmap=poynting_cmap,
                      norm=poynting_norm, scale=resolution, headwidth=5)
        
        sm = cm.ScalarMappable(cmap=poynting_cmap, norm=poynting_norm)
        cb = plt.colorbar(sm, ax=ax)
        cb.ax.set_xlabel(r'\|\,\mathbf{S}\,\|$')

        if color_obj is not None:
            draw_object_2d(ax, *self.solver.objects, color=color_obj)

        set_axis_labels_2d(ax, axis_slice)
        ax.set_title('Energy Flow')
        ax.set_aspect(self.aspect)     
    
    def get_field_varables(self, field_name):
        variables = [self.get_field_from_str(field_name)]
        index = self.fields.index(field_name)

        for variable in (self.field_colors, self.object_colors,
                         self.cmaps, self.norms):
            if isinstance(variable, str):
                new_variable = variable
            else:
                new_variable = variable[index]

            if variable == self.cmaps:
                new_variable = get_cmap_if_str(new_variable)

            variables.append(new_variable)

        return tuple(variables)
    
    def get_field_from_str(self, field_name):
        if field_name.upper() == 'E':
            return self.solver.E
        if field_name.upper() == 'H':
            return self.solver.H
        if field_name.upper() == 'S':
            return self.solver.get_poynting()
        if field_name.upper() == 'U':
            return self.solver.get_energy()
    
    def get_fig_and_axs(self, dim, field_num):
        if self.orientation == 'v':
            return plt.subplots(field_num, dim, figsize=self.figsize)
        elif self.orientation == 'h':
            return plt.subplots(dim, field_num, figsize=self.figsize)
        elif self.orientation == 'c':
            if dim == field_num:
                return plt.subplots(dim, dim, figsize=self.figsize)
            elif dim == 1 and field_num % 2 == 0:
                fig, axs = plt.subplots(int(field_num / 2), int(field_num / 2),
                                        figsize=self.figsize)
                return fig, axs.flatten()
            elif dim == 1 and field_num == 3:
                fig = plt.figure(figsize=self.figsize)
                gs = gridspec.GridSpec(4, 4)
                axs = np.array((plt.subplot(gs[:2, :2]),
                                plt.subplot(gs[:2, 2:]),
                                plt.subplot(gs[2:, 1:3])))
                return fig, axs
            elif dim == 3 and field_num == 2:
                fig = plt.figure(figsize=self.figsize)
                gs = gridspec.GridSpec(7, 11)
                axs = np.array(
                    (plt.subplot(gs[:3, 3:5]), plt.subplot(gs[4:, 3:5]),
                     plt.subplot(gs[2:5, :2]), plt.subplot(gs[:3, 6:8]),
                     plt.subplot(gs[4:, 6:8]), plt.subplot(gs[2:5, 9:]))
                )
                axs = np.reshape(axs, (2, 3))
                return fig, axs
    
    def get_data_1d(self, field, axis_space, axis_field,
                    slice_first_coordinate, slice_second_coordinate):
        begin_cell, end_cell = \
            self.begin_cell[axis_space], self.end_cell[axis_space]
        slice_first_cell, slice_second_cell = (
            int(slice_coordinate / self.solver.grid_dist) for slice_coordinate
            in (slice_first_coordinate, slice_second_coordinate)
        )
        
        data_space = np.linspace(self.begin_pos[axis_space],
                                 self.end_pos[axis_space],
                                 self.cell_count[axis_space])

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
        if len(field.shape) == 3:
            slices.pop()
        return field[tuple(slices)]
    
    def get_data_space_2d(self, axis_slice):
        axis_space = list(range(3))
        axis_space.remove(axis_slice)
        return (
            np.linspace(self.begin_pos[axis_space[0]],
                        self.end_pos[axis_space[0]],
                        self.cell_count[axis_space[0]]),
            np.linspace(self.begin_pos[axis_space[1]],
                        self.end_pos[axis_space[1]],
                        self.cell_count[axis_space[1]])
        )
    
    def get_bounds_2d(self, axis_slice):
        axis_space = list(range(3))
        axis_space.remove(axis_slice)
        return (self.begin_pos[axis_space[0]], self.end_pos[axis_space[0]],
                self.begin_pos[axis_space[1]], self.end_pos[axis_space[1]])
    
    def reduce_data(self, data_space, data_field, resolution=25):
        dim = len(data_space)
        index_min = min(range(dim), key=lambda i : len(data_space[i]))

        index_reduced = tuple(
            np.round(
                np.linspace(0, len(data_space[i]) - 1,
                            int(round(resolution * len(data_space[i])
                                      / len(data_space[index_min]))))
            ).astype(int)
            for i in range(dim)
        )

        data_space = tuple(data_space[i][index_reduced[i]] for i in range(dim))
        data_field = data_field[
            np.meshgrid(*index_reduced, np.array(tuple(range(dim))))
        ]
        return data_space, data_field
    
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
        
        self.cell_count = tuple(end_c - begin_c for (begin_c, end_c)
                                in zip(self.begin_cell, self.end_cell))
    
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
                lw=0, alpha=0.17, color=color
            ))


def get_norm_if_str(norm, vmin, vmax):
    if isinstance(norm, str):
        if norm.lower() in ('lin', 'linear'):
            return colors.Normalize(vmin=vmin, vmax=vmax)
        elif norm.lower() in ('log', 'logarithmic'):
            if abs(vmax) < 1e-9:
                linthresh = 1
            elif vmin == 0:
                linthresh = 10 ** (round(log10(abs(vmax))) - 5)
            else:
                linthresh = max(10 ** (round(log10(abs(vmin)))),
                                10 ** (round(log10(abs(vmax))) - 5))
            return colors.SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
    else:
        return norm


def get_cmap_if_str(cmap):
    if isinstance(cmap, str):
        cmap = copy(cm.get_cmap(cmap))
        cmap.set_bad(cmap(0))
    return cmap


def get_axis_name(axis_num):
    axes = ('x', 'y', 'z')
    return axes[axis_num]


def get_field_label(field_name, axis_field):
    return '$' + field_name + '_' + get_axis_name(axis_field) + '$'


def set_axis_labels_2d(ax, axis_slice):
    axis_space = list(range(3))
    axis_space.pop(axis_slice)
    ax.set_xlabel('$' + get_axis_name(axis_space[0]) + '$')
    ax.set_ylabel('$' + get_axis_name(axis_space[1]) + '$')
