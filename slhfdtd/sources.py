from math import pi, sin, floor
import numpy as np

from .solving import SPEED_LIGHT


class Source:
    def __init__(self, begin, end, direction=2,
                 additive=True, func=sin, power=1.0,
                 wavelength=None, freq=None, phase=0.0):
        self.begin_pos, self.end_pos = begin, end
        self.direction, self.additive = direction, additive
        self.power, self.phase, self.func = power, phase, func

        if wavelength is not None:
            self.wavelength = wavelength
            if wavelength == 0:
                self.omega = 0
            else:
                self.omega = 2*pi * SPEED_LIGHT/self.wavelength
        elif freq is not None:
            if freq == 0:
                self.wavelength = 0
            else:
                self.wavelength = SPEED_LIGHT/freq
            self.omega = 2*pi * freq
        else:
            self.wavelength = 300e-9
            self.omega = 2*pi * SPEED_LIGHT/self.wavelength

        self.current_time_step = 0

    def set_solver(self, solver):
        self.solver = solver
        self.set_pos()
        self.set_amplitude()

    def set_pos(self):
        self.begin_cell = list(round(begin / self.solver.grid_dist)
                               for begin in self.begin_pos)
        self.end_cell = list(round(end / self.solver.grid_dist)
                             for end in self.end_pos)
        for i in range(3):
            if self.end_cell[i] == self.begin_cell[i]:
                self.end_cell[i] += 1

        self.pos = (
            *(slice(begin_c, end_c)
              for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)),
            int(self.direction) if self.direction is not None
            else slice(None)
        )

    def set_amplitude(self):
        self.amplitude = (
            self.power / self.solver.permittivity[self.pos]
        )**0.5

    def step(self):
        self.update_E()
        self.update_H()
        self.current_time_step += 1

    def update_E(self):
        if self.additive:
            self.solver.E[self.pos] += (
                self.amplitude * self.func(
                    self.omega
                    * self.current_time_step * self.solver.time_step
                    + self.phase
                )
            )
        else:
            self.solver.E[self.pos] = (
                self.amplitude * self.func(
                    self.omega * self.current_time_step
                    * self.solver.time_step + self.phase
                )
            )

    def update_H(self):
        pass


class PointSource(Source):
    def __init__(self, pos, direction=2, additive=True, func=sin,
                 power=1.0, wavelength=None, freq=None, phase=0.0):
        super().__init__(pos, pos, direction, additive, func,
                         power, wavelength, freq, phase)


class LineSource(Source):
    def set_pos(self):
        self.begin_cell = list(round(begin / self.solver.grid_dist)
                               for begin in self.begin_pos)
        self.end_cell = list(round(end / self.solver.grid_dist)
                             for end in self.end_pos)

        length = int(sum((end_c - begin_c)**2 for (begin_c, end_c)
                         in zip(self.begin_cell, self.end_cell))**0.5)

        self.pos = tuple(
            np.ones((length,)).astype(np.int) * int(begin_c)
            if begin_c == end_c
            else np.linspace(begin_c, end_c, length).astype(np.int)

            for (begin_c, end_c) in zip(self.begin_cell, self.end_cell)
        ) + (self.direction,)


def square_wave(theta):
    if floor(theta/pi) % 2 == 0:
        return 1
    else:
        return -1
