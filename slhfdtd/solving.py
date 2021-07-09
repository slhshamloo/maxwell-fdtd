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


def get_shape(length_x, length_y, length_z, grid_dist=1e-10):
        return (round(length_x/grid_dist) + 1,
                round(length_y/grid_dist) + 1,
                round(length_z/grid_dist) + 1)


class Solver:
    def __init__(self, length_x, length_y, length_z,
                 grid_dist=1e-10, courant_number=None,
                 permittivity=1.0, permeability=1.0,
                 init_E=None, init_H=None):
        self.grid_dist = grid_dist
        
        self.length_x, self.length_y, self.length_z = (
            length_x, length_y, length_z)
        self.cell_count_x, self.cell_count_y, self.cell_count_z = get_shape(
            length_x, length_y, length_z, grid_dist)

        dim = int(self.cell_count_x > 0) + \
            int(self.cell_count_y > 0) + int(self.cell_count_z > 0)

        self.courant_number = (courant_number if courant_number is not None
            else 0.9999 * dim**(-0.5))

        self.time_step = grid_dist * self.courant_number / SPEED_LIGHT
        self.current_time_step = 0
        
        self.set_permittivity(permittivity)
        self.set_permeability(permeability)

        self.E = (init_E if init_E is not None else np.zeros(
            (self.cell_count_x, self.cell_count_y, self.cell_count_z, 3)))
        self.H = (init_H if init_H is not None else np.zeros(
            (self.cell_count_x, self.cell_count_y, self.cell_count_z, 3)))
        
        self.sources = []
        
    def set_permittivity(self, permittivity):
        if type(permittivity) is np.ndarray:
            permittivity = permittivity[:, :, :, None]
        
        self.inverse_permittivity = (np.ones(
            (self.cell_count_x, self.cell_count_y, self.cell_count_z, 3))
            / permittivity)
    
    def set_permeability(self, permeability):
        if type(permeability) is np.ndarray:
            permeability = permeability[:, :, :, None]
        self.inverse_permeability = (np.ones(
            (self.cell_count_x, self.cell_count_y, self.cell_count_z, 3))
            / permeability)
    
    def add_source(self, source):
        source.set_solver(self)
        self.sources.append(source)
    
    def update_E(self):
        constant = self.courant_number * VACUUM_IMPEDANCE
        self.E += constant * self.inverse_permittivity * curl_H(self.H)
    
    def update_H(self):
        constant = self.courant_number / VACUUM_IMPEDANCE
        self.H -= constant * self.inverse_permeability * curl_E(self.E)
    
    def step(self):
        for source in self.sources:
            source.step()
        self.update_E()
        self.update_H()
        self.current_time_step += 1
    
    def run(self, time):
        for _ in range(round(time/self.time_step)):
            self.step()
