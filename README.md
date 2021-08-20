# Maxwell FDTD
Maxwell's equations solver using the finite-difference time-domain method.  
Electromagnetism 2 course project.

The initial project deadline was June 8th. I couldn't implement all the features I wanted in the narrow time window I worked on the project before the deadline, and I also enjoyed it, so I continued working on it during the summer.

## Description
My goal for this project was to implement a simple FDTD solver, simulate some intersting electrodynamic systems, and write a paper roughly describing the fundamentals of FDTD simulation and presenting my work.

There are many great FDTD solvers out there. In particular, Floris Laporte's `fdtd` package ([flaport/fdtd](https://github.com/flaport/fdtd.git); Check it out, it's great!) is completely written in python and has most of the features needed for the simulations in this project. So I decided to implement a simpler version of it and add any other features I needed.

## Setup
```sh
python setup.py install
```
