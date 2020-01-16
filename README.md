# hbox-examples
H-box method for Shallow Water Examples - 1D

This directory contains a number of examples for solving the shallow
water equations in one dimension using hbox method. Please download solverMB.py file under cr2940/pyclaw and rename to solver.py (and original solver.py to something else) to run the zero-width barrier simulations.

Note that many of these scripts were set up to make nice movies and output 
much more often than is probably necessary.

Examples
========
 - Well-Balancing and Ghost Fluid (sill_edge.py, shallow...redistribute.py): Test which uses a setting of 
   a steady state including a wall assigning at the edge.
 - Conservation of mass (mass_conservation.ipynb): Demonstration that the 
   wave redistribution method maintains conservation. Includes a jump in depth at 
   x = -0.2 with zero momentum, which is the initial condition of the classic 
   dam-break problem.
 - Leaky Barriers (sill_h_box_wave.py, shallow...wave_MB.py): Test that the barrier off edge does indeed keep
   water from flowing past it provided that the barrier is high enough. (Adjust wall_height in parameters_h_box_wave.txt)
 - Over-Topped Barrier on a Sloping Beach (sill_h_box_wave.py): Example of coastal flood 
   modeling with a sloping bathymetry and wet-dry interface. In this case the incoming 
   wave has enough momentum so that the wave overcomes the barrier and leads to flooding  
   on the other side of the barrier. 
