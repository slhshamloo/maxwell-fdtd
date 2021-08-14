import numpy as np


class Boundary:
    def __init__(self, begin, end):
        self.begin_pos = begin
        self.end_pos = end

    def set_solver(self, solver):
        self.set_pos(solver.grid_dist)
        self.solver = solver

    def set_pos(self, grid_dist):
        self.begin_cell = list(round(begin / grid_dist)
                               for begin in self.begin_pos)
        self.end_cell = list(round(end / grid_dist)
                             for end in self.end_pos)
        for i in range(3):
            if self.end_cell[i] == self.begin_cell[i]:
                self.end_cell[i] += 1

        self.shape = (
            *(end_c - begin_c
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            3
        )
        self.pos = (
            *(slice(begin_c, end_c)
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            slice(None)
        )

    def update_E_before(self):
        pass

    def update_H_before(self):
        pass

    def update_E_after(self):
        pass

    def update_H_after(self):
        pass


class Reflector(Boundary):
    def update_E_after(self):
        self.solver.E[self.pos] = np.zeros(self.shape)

    def update_H_after(self):
        self.solver.H[self.pos] = np.zeros(self.shape)


class AutoReflector(Boundary):
    def __init__(self):
        pass

    def set_solver(self, solver):
        if solver.cell_count[0] > 1:
            solver.add_boundary(Reflector((0, 0, 0),
                (solver.grid_dist, solver.length[1], solver.length[2])
            ))
            solver.add_boundary(Reflector(
                (solver.length[0] - solver.grid_dist, 0, 0),
                solver.length
            ))
        elif solver.cell_count[1] > 1:
            solver.add_boundary(Reflector((0, 0, 0),
                (solver.length[0], solver.grid_dist, solver.length[2])
            ))
            solver.add_boundary(Reflector(
                (0, solver.length[1] - solver.grid_dist, 0),
                solver.length
            ))
        elif solver.cell_count[2] > 1:
            solver.add_boundary(Reflector((0, 0, 0),
                (solver.length[0], solver.length[1], solver.grid_dist)
            ))
            solver.add_boundary(Reflector(
                (0, 0, solver.length[2] - solver.grid_dist),
                solver.length
            ))


class PML(Boundary):
    def __init__(self, begin, end, direction, reverse=False,
                 scaling_factor=1.0, stability_factor=1e-8):
        super().__init__(begin, end)
        self.direction, self.scaling_factor, self.stability_factor = \
            direction, scaling_factor, stability_factor
        self.reverse = reverse

    def set_solver(self, solver):
        super().set_solver(solver)
        self.set_absorption()
        self.initialize_field_parameters()

    def set_absorption(self):
        if self.reverse:
            absorption_profile_E = np.arange(
                self.end_cell[self.direction]
                - self.begin_cell[self.direction] + 0.5,
                0.5,
                -1.0
            )
            absorption_profile_H = np.arange(
                self.end_cell[self.direction]
                - self.begin_cell[self.direction],
                0.0,
                -1.0
            )
        else:
            absorption_profile_E = np.arange(
                0.5, self.end_cell[self.direction]
                - self.begin_cell[self.direction] + 0.5,
                1.0
            )
            absorption_profile_H = np.arange(
                0.0, self.end_cell[self.direction]
                - self.begin_cell[self.direction],
                1.0
            )

        sigma_E = np.zeros(self.shape)
        sigma_H = np.zeros(self.shape)

        sigma1d_E = \
            40 * absorption_profile_E**3 / len(absorption_profile_E)**4
        sigma1d_H = \
            40 * absorption_profile_H**3 / len(absorption_profile_H)**4

        if self.direction == 0:
            sigma_E[:, :, :, 0] = sigma1d_E[:, None, None]
            sigma_H[:, :, :, 0] = sigma1d_H[:, None, None]
        elif self.direction == 1:
            sigma_E[:, :, :, 1] = sigma1d_E[None, :, None]
            sigma_H[:, :, :, 1] = sigma1d_H[None, :, None]
        else:
            sigma_E[:, :, :, 2] = sigma1d_E[None, None, :]
            sigma_H[:, :, :, 2] = sigma1d_H[None, None, :]

        self.b_E = np.exp(
            -(self.stability_factor + sigma_E / self.scaling_factor)
            * self.solver.courant_number
        )
        self.c_E = (
            (self.b_E - 1) * sigma_E
            / (sigma_E * self.scaling_factor
               + self.stability_factor * self.scaling_factor**2)
        )

        self.b_H = np.exp(
            -(self.stability_factor + sigma_H / self.scaling_factor)
            * self.solver.courant_number
        )
        self.c_H = (
            (self.b_H - 1) * sigma_H
            / (sigma_H * self.scaling_factor
               + self.stability_factor * self.scaling_factor**2)
        )

    def initialize_field_parameters(self):
        self.phi_E = np.zeros(self.shape)
        self.phi_H = np.zeros(self.shape)
        self.psi_Ex = np.zeros(self.shape)
        self.psi_Ey = np.zeros(self.shape)
        self.psi_Ez = np.zeros(self.shape)
        self.psi_Hx = np.zeros(self.shape)
        self.psi_Hy = np.zeros(self.shape)
        self.psi_Hz = np.zeros(self.shape)

    def update_E_before(self):
        b = self.b_E
        c = self.c_E

        self.psi_Ex *= b
        self.psi_Ey *= b
        self.psi_Ez *= b

        Hx = self.solver.H[self.pos[:-1] + (0,)]
        Hy = self.solver.H[self.pos[:-1] + (1,)]
        Hz = self.solver.H[self.pos[:-1] + (2,)]

        self.psi_Ex[:, 1:, :, 1] += \
            (Hz[:, 1:, :] - Hz[:, :-1, :]) * c[:, 1:, :, 1]
        self.psi_Ex[:, :, 1:, 2] += \
            (Hy[:, :, 1:] - Hy[:, :, :-1]) * c[:, :, 1:, 2]

        self.psi_Ey[:, :, 1:, 2] += \
            (Hx[:, :, 1:] - Hx[:, :, :-1]) * c[:, :, 1:, 2]
        self.psi_Ey[1:, :, :, 0] += \
            (Hz[1:, :, :] - Hz[:-1, :, :]) * c[1:, :, :, 0]

        self.psi_Ez[1:, :, :, 0] += \
            (Hy[1:, :, :] - Hy[:-1, :, :]) * c[1:, :, :, 0]
        self.psi_Ez[:, 1:, :, 1] += \
            (Hx[:, 1:, :] - Hx[:, :-1, :]) * c[:, 1:, :, 1]

        self.phi_E[..., 0] = self.psi_Ex[..., 1] - self.psi_Ex[..., 2]
        self.phi_E[..., 1] = self.psi_Ey[..., 2] - self.psi_Ey[..., 0]
        self.phi_E[..., 2] = self.psi_Ez[..., 0] - self.psi_Ez[..., 1]

    def update_H_before(self):
        b = self.b_H
        c = self.c_H

        self.psi_Hx *= b
        self.psi_Hy *= b
        self.psi_Hz *= b

        Ex = self.solver.E[self.pos[:-1] + (0,)]
        Ey = self.solver.E[self.pos[:-1] + (1,)]
        Ez = self.solver.E[self.pos[:-1] + (2,)]

        self.psi_Hx[:, :-1, :, 1] += \
            (Ez[:, 1:, :] - Ez[:, :-1, :]) * c[:, :-1, :, 1]
        self.psi_Hx[:, :, :-1, 2] += \
            (Ey[:, :, 1:] - Ey[:, :, :-1]) * c[:, :, :-1, 2]

        self.psi_Hy[:, :, :-1, 2] += \
            (Ex[:, :, 1:] - Ex[:, :, :-1]) * c[:, :, :-1, 2]
        self.psi_Hy[:-1, :, :, 0] += \
            (Ez[1:, :, :] - Ez[:-1, :, :]) * c[:-1, :, :, 0]

        self.psi_Hz[:-1, :, :, 0] += \
            (Ey[1:, :, :] - Ey[:-1, :, :]) * c[:-1, :, :, 0]
        self.psi_Hz[:, :-1, :, 1] += \
            (Ex[:, 1:, :] - Ex[:, :-1, :]) * c[:, :-1, :, 1]

        self.phi_H[..., 0] = self.psi_Hx[..., 1] - self.psi_Hx[..., 2]
        self.phi_H[..., 1] = self.psi_Hy[..., 2] - self.psi_Hy[..., 0]
        self.phi_H[..., 2] = self.psi_Hz[..., 0] - self.psi_Hz[..., 1]

    def update_E_after(self):
        self.solver.E[self.pos] += (
            self.solver.constant_E
            / self.solver.permittivity[self.pos]
            * self.phi_E
        )

    def update_H_after(self):
        self.solver.H[self.pos] -= (
            self.solver.constant_H
            / self.solver.permeability[self.pos]
            * self.phi_H
        )


class AutoPML(Boundary):
    def __init__(self, thickness=None, is_thickness_cell_count=False,
                 scaling_factor=1.0, stability_factor=1e-8):
        self.is_thickness_cell_count = is_thickness_cell_count
        self.thickness, self.scaling_factor, self.stability_factor = (
            thickness, scaling_factor, stability_factor)

    def set_solver(self, solver):
        if self.thickness is None:
            if len(solver.sources) > 0:
                self.thickness = max(
                    source.wavelength for source in solver.sources
                )
            else:
                self.thickness = 0

            if self.thickness == 0:
                self.thickness = 10
                self.is_thickness_cell_count = True
            else:
                self.thickness = min(
                    self.thickness,
                    np.min(
                        np.asarray(solver.length)[np.nonzero(solver.length)]
                    ) / 6
                )
        if self.is_thickness_cell_count:
            self.thickness = self.thickness * solver.grid_dist

        self.begin_bound = 3 * [0]
        self.end_bound = list(solver.length)

        if solver.length[0] > 2 * self.thickness:
            solver.add_boundary(PML(
                (0, 0, 0),
                (self.thickness, solver.length[1], solver.length[2]),
                0, True, self.scaling_factor, self.stability_factor
            ))
            solver.add_boundary(PML(
                (solver.length[0] - self.thickness, 0, 0),
                solver.length, 0, False,
                self.scaling_factor, self.stability_factor
            ))

            self.begin_bound[0] = self.thickness
            self.end_bound[0] = solver.length[0] - self.thickness

        if solver.length[1] > 2 * self.thickness:
            solver.add_boundary(PML(
                (0, 0, 0),
                (solver.length[0], self.thickness, solver.length[2]),
                1, True, self.scaling_factor, self.stability_factor
            ))
            solver.add_boundary(PML(
                (0, solver.length[1] - self.thickness, 0),
                solver.length, 1, False,
                self.scaling_factor, self.stability_factor
            ))

            self.begin_bound[1] = self.thickness
            self.end_bound[1] = solver.length[1] - self.thickness

        if solver.length[2] > 2 * self.thickness:
            solver.add_boundary(PML(
                (0, 0, 0),
                (solver.length[0], solver.length[1], self.thickness),
                2, True, self.scaling_factor, self.stability_factor
            ))
            solver.add_boundary(PML(
                (0, 0, solver.length[2] - self.thickness),
                solver.length, 2, False,
                self.scaling_factor, self.stability_factor
            ))

            self.begin_bound[2] = self.thickness
            self.end_bound[2] = solver.length[2] - self.thickness

class Exact1DAbsorber(Boundary):
    def __init__(self, direction):
        self.direction = direction

        self.begin, self.begin_match, self.end, self.end_match = (
            4 * [slice(None)] for _ in range(4)
        )

        self.begin[direction] = 0
        self.begin_match[direction] = 1
        self.end[direction] = -1
        self.end_match[direction] = -2

        self.begin, self.begin_match, self.end, self.end_match = (
            tuple(self.begin), tuple(self.begin_match),
            tuple(self.end), tuple(self.end_match)
        )

        self.step_E = False
        self.step_H = False

    def set_solver(self, solver):
        self.solver = solver

    def update_E_before(self):
        if self.step_E:
            self.solver.E[self.begin] = self.solver.E[self.begin_match]
        self.step_E = not self.step_E

    def update_H_before(self):
        if self.step_H:
            self.solver.H[self.end] = self.solver.H[self.end_match]
        self.step_H = not self.step_H
