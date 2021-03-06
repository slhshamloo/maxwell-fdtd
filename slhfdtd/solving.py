import numpy as np


SPEED_LIGHT = 299_792_458.0
VACUUM_PERMEABILITY = 4e-7 * np.pi
VACUUM_PERMITTIVITY = 1/(VACUUM_PERMEABILITY * SPEED_LIGHT**2)
VACUUM_IMPEDANCE = VACUUM_PERMEABILITY * SPEED_LIGHT


def curl_E(E):
    curl = np.zeros(E.shape)

    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl


def curl_H(H):
    curl = np.zeros(H.shape)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl


class Solver:
    def __init__(self, length, grid_dist=1e-8, courant_number=None,
                 permittivity=1.0, permeability=1.0, conductivity=0.0,
                 init_E=None, init_H=None):
        self.grid_dist = grid_dist
        self.length = length
        self.cell_count = [round(length_axis/grid_dist)
                           for length_axis in length]
        for i in range(len(self.cell_count)):
            if self.cell_count[i] == 0:
                self.cell_count[i] = 1

        dim = sum(int(c_count > 0) for c_count in self.cell_count)

        self.courant_number = (courant_number if courant_number is not None
                               else 0.9999 * dim**(-0.5))

        self.time_step = grid_dist * self.courant_number / SPEED_LIGHT
        self.current_time_step = 0

        self.set_permittivity(permittivity)
        self.set_permeability(permeability)
        self.set_conductivity(conductivity)

        self.constant_E = self.courant_number * VACUUM_IMPEDANCE
        self.constant_H = self.courant_number / VACUUM_IMPEDANCE

        self.E = (init_E if init_E is not None
                  else np.zeros((*self.cell_count, 3)))
        self.H = (init_H if init_H is not None
                  else np.zeros((*self.cell_count, 3)))

        self.sources = []
        self.objects = []
        self.boundaries = []

    def set_permittivity(self, permittivity):
        if type(permittivity) is np.ndarray:
            permittivity = permittivity[:, :, :, None]
        self.permittivity = np.ones((*self.cell_count, 3)) * permittivity

    def set_permeability(self, permeability):
        if type(permeability) is np.ndarray:
            permeability = permeability[:, :, :, None]
        self.permeability = np.ones((*self.cell_count, 3)) * permeability

    def set_conductivity(self, conductivity):
        if type(conductivity) is np.ndarray:
            conductivity = conductivity[:, :, :, None]
        dissipation = (np.ones((*self.cell_count, 3))
                       * 0.5 * conductivity / self.permittivity
                       / VACUUM_PERMITTIVITY)
        self.dissipation_mult = (1 - dissipation) / (1 + dissipation)
        self.dissipation_add = 1 / (1 + dissipation)
    
    def get_poynting(self):
        return np.cross(self.E, self.H)
    
    def get_energy(self):
        return ((np.sum(self.E**2 * self.permittivity, axis=3)
                 * VACUUM_PERMITTIVITY
                 + np.sum(self.H**2 / self.permeability, axis=3)
                 / VACUUM_PERMEABILITY)
                / 2)

    def add_source(self, source):
        source.set_solver(self)
        self.sources.append(source)

    def add_object(self, obj):
        obj.set_solver(self)
        self.objects.append(obj)

    def add_boundary(self, boundary):
        boundary.set_solver(self)
        self.boundaries.append(boundary)

    def update_E(self):
        for boundary in self.boundaries:
            boundary.update_E_before()

        self.E *= self.dissipation_mult
        self.E += (self.dissipation_add * self.constant_E
                   / self.permittivity * curl_H(self.H))

        for boundary in self.boundaries:
            boundary.update_E_after()

    def update_H(self):
        for boundary in self.boundaries:
            boundary.update_H_before()

        self.H -= self.constant_H / self.permeability * curl_E(self.E)

        for boundary in self.boundaries:
            boundary.update_H_after()

    def step(self):
        for source in self.sources:
            source.step()

        self.update_E()
        self.update_H()

        self.current_time_step += 1

    def run(self, time):
        for _ in range(round(time/self.time_step)):
            self.step()
