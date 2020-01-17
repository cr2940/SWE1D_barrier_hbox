# hbox-examples
H-box method for Shallow Water Examples - 1D

These codes solve the 1d shallow water equations (SWE) with zero width barrier using GeoClaw augmented solver and "double h-boxes" method. One can simulate a barrier by introducing a step like jump in bathymetry, but if we want a thin barrier, then the grid size of the bathymetry jump must also be very thin, which then requires more resolution/computation. We avoid this by using an h-box method and wave redistribution. (Read nonLTS1D.pdf https://github.com/cr2940/SWE1D_barrier_hbox/blob/master/nonLTS1D.pdf for descriptive notes.)

Please download solverMB.py file under cr2940/pyclaw and rename to solver.py (and original solver.py to something else) to run the zero-width barrier simulations.

Note that many of these scripts were set up to make nice movies and output 
much more often than is probably necessary.

Examples
========
 - Simulation examples: all files that start with sill_....py are setup files for specific simulations.
 - Distinguishing '...cellwall' vs '...highwall': the cellwall files model the barrier as bathymetry jump with single cell-width while highwall files are the zero-width wall and use the h-box method. 
 - Well-Balancing and Ghost Fluid (sill_edge.py, shallow...redistribute.py): Test which uses a setting of 
   a steady state including a wall assigning at the edge.
 - Conservation of mass (mass_conservation.ipynb): Demonstration that the 
   wave redistribution method maintains conservation. Includes a jump in depth at 
   x = -0.2 with zero momentum, which is the initial condition of the classic 
   dam-break problem. (only for wall on edge)
