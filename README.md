# hbox-examples
H-box method for Shallow Water Examples - 1D

These codes solve the 1d shallow water equations (SWE) with zero width barrier using GeoClaw augmented solver and "double h-boxes" method. One can simulate a barrier by introducing a step like jump in bathymetry, but if we want a thin barrier, then the grid size of the bathymetry jump must also be very thin, which then requires more resolution/computation. We avoid this by using an h-box method and wave redistribution. (Read nonLTS1D.pdf https://github.com/cr2940/SWE1D_barrier_hbox/blob/master/nonLTS1D.pdf for descriptive notes.)

Please download solver.py file under cr2940/pyclaw to run the zero-width barrier simulations.

Examples
========
 - Simulation examples: all files that start with sill_....py are setup files for specific simulations.
 - Wall on edge (sill_edge.py, shallow...redistribute.py): These are simulations where barrier/wall is on edge of grid cell
 - See Clawpack website (clawpack.org) and Pyclaw on how to make your own simulation (modifying setup files)
