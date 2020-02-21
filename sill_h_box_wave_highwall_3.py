#!/usr/bin/env python
# encoding: utf-8

r"""
Shallow water flow with zero-width barrier slightly off grid edge by alpha*dx
======================================================================
Solve the one-dimensional shallow water equations including bathymetry:
.. math::
    h_t + (hu)_x & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x & = -g h b_x.
Here h is the depth, u is the velocity, g is the gravitational constant, and b
the bathymetry.
"""
import sys
import numpy
import matplotlib.pyplot as plt
from clawpack import riemann
import shallow_1D_redistribute_wave_MB
from clawpack.pyclaw.plot import plot

def before_step(solver, states):
    drytol = states.problem_data['dry_tolerance']
    for i in range(len(states.q[0,:])):
        states.q[0,i] = max(states.q[0,i], 0.0)
        if states.q[0,i] < drytol:
            states.q[1,i] = 0.0
    return states

def load_parameters(fileName):
    fileObj = open(fileName)
    params = {}
    for line in fileObj:
        line = line.strip()
        key_value = line.split('=')
        params[key_value[0]] = key_value[1]
    return params

def hbox_bc(state,dim,t,qbc,auxbc,num_ghost=2):
    alpha = 0.5
    qbc[:,1] = alpha*qbc[:,2] + (1-alpha)*qbc[:,3]
    qbc[1,1] *= -1
    qbc[:,0] = qbc[:,1]

    qbc[:,-2] = alpha*qbc[:,-4] + (1-alpha)*qbc[:,-3]
    qbc[1,-2] *= -1
    qbc[:,-1] = qbc[:,-2]

    auxbc[0,0] = alpha*auxbc[0,2] + (1-alpha)*auxbc[0,3]
    auxbc[0,1] = auxbc[0,0]
    auxbc[1,0] = auxbc[1,3]
    auxbc[1,1] = auxbc[1,0]
    auxbc[0,-1] = alpha*auxbc[0,-4] + (1-alpha)*auxbc[0,-3]
    auxbc[0,-2] = auxbc[0,-1]
    auxbc[1,-1] = auxbc[1,-4]
    auxbc[1,-2] = auxbc[1,-1]

def setup(kernel_language='Python',use_petsc=False, outdir='./_output3_4', solver_type='classic'):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    params = load_parameters('parameters_h_box_wave.txt')
    xlower = float(params['xlower'])
    xupper = float(params['xupper'])
    cells_number = int(params['cells_number'])
    nw = int(params['wall_position']) # index of the edge used to present the wall
    alpha = float(params['fraction'])
    wall_height = float(params['wall_height'])

    x = pyclaw.Dimension(xlower, xupper, cells_number, name='x')
    del_xp = (xupper-xlower)/(cells_number-1)
    del_xc = (xupper-xlower)/(cells_number)

    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2, 2)
    # shifting the domain to physical setup
    xc = state.grid.x.centers
    for m in range(len(xc)):
        if m < nw-1:
            xc[m] = (2*m+1)/2 * del_xp
        if m == nw-1:
            xc[m] = (m+(1/2)*alpha) * del_xp
        if m == nw:
            xc[m] = (m+(1/2)*(1+alpha)) * del_xp
        if m >= nw+1:
            xc[m] = (2*(m-1)+1)/2 * del_xp

    xn = state.grid.x.nodes
    for k in range(len(xn)):
        if k < nw:
            xn[k] = k*del_xp
        if k == nw:
            xn[k] = (k+alpha)*del_xp
        if k > nw:
            xn[k] = (k-1) * del_xp

    # Gravitational constant
    state.problem_data['grav'] = 9.8
    state.problem_data['sea_level'] = -0.4

    # Wall position
    state.problem_data['wall_position'] = nw
    state.problem_data['wall_height'] = wall_height
    state.problem_data['fraction'] = alpha
    state.problem_data['dry_tolerance'] = 0.001
    state.problem_data['max_iteration'] = 4
    state.problem_data['method'] = 'h_box'
    state.problem_data['zero_width'] = True
    state.problem_data['xupper'] = xupper
    state.problem_data['xlower'] = xlower
    state.problem_data['cells_num'] = cells_number


    solver = pyclaw.ClawSolver1D(shallow_1D_redistribute_wave_MB.shallow_fwave_hbox_dry_1d)

    solver.limiters = pyclaw.limiters.tvd.minmod
    solver.order = 1
    solver.cfl_max = 0.9
    solver.cfl_desired = 0.8
    solver.kernel_language = "Python"
    solver.fwave = True
    solver.num_waves = 3
    solver.num_eqn = 2
    solver.before_step = before_step
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.aux_bc_lower[0] = pyclaw.BC.wall
    solver.aux_bc_upper[0] = pyclaw.BC.wall

    # Initial Conditions
    xpxc = (cells_number) * 1.0 / (cells_number-1)
    state.problem_data['xpxc'] = xpxc
    state.aux[0, :] = - 1.2 * numpy.ones(xc.shape)
    # state.aux[0, nw:] = -0.6
    # the wall height was 1.1 for this exmaple
    # state.aux[0, 100:] = -0.8
    #state.aux[0,nw:nw+2] = -0.1

    ## slope bathymetry
    # bathymetry = numpy.linspace(-0.8, -0.4, xc.shape[0] - 1, endpoint=True)
    # state.aux[0,:nw-1] = bathymetry[:nw-1]
    # state.aux[0,nw-1] = bathymetry[nw-1]
    # state.aux[0,nw:] = bathymetry[nw-1:]
#    state.aux[0, :] = numpy.linspace(-0.8, -0.4, xc.shape[0], endpoint=True)
    state.aux[1,:] = numpy.zeros(xc.shape)
    state.aux[1, :] = xpxc # change this to actual delta x_p and xp is actuallly 1/N-1
    # # shifting the grid such that barrier aligns with cell edge nw and pushing the small cells to the endpoint boundaries
    # so will need two hbox pairs: one at left endpoint and one at right endpoint
    state.aux[1, nw-1] = alpha * xpxc
    state.aux[1, nw] = (1 - alpha) * xpxc
    state.q[0, :] = 0.0 - state.aux[0, :]
    # state.q[0,320:] += 0.5
    # state.q[0,:80] += 0.5
    state.q[0,:] = state.q[0,:].clip(min=0)
    # state.q[0,nw:] = 0.0 # uncomment for dry state on right side of wall
    state.q[1,:nw] = 0.4
    state.q[1,nw:] = 0.5
    print(state.q[0,:])


    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.tfinal = 0.70
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    # claw.setplot = setplot
    claw.write_aux_init = True

    dx = claw.solution.domain.grid.delta[0]

    state.mF=1
    init_mass = print(numpy.sum(state.q[0,:]*(1/cells_number)*state.aux[1,:],axis=0))
    def mass_change(state):
        #print(state.q[0,:])
        state.F[0,:] = numpy.sum(state.q[0,:]*(1/cells_number)*state.aux[1,:],axis=0)
    claw.compute_F = mass_change
    claw.F_file_name = 'total_mass'


    claw.output_style = 1
    claw.num_output_times = 20
    claw.nstepout = 1

    # if outdir is not None:
    #     claw.outdir = outdir
    # else:
    #     claw.output_format = None

    claw.outdir = outdir
    if outdir is None:
        claw.output_format = None

    claw.run()
#    return claw

    # check conservation:
# get the solution q and capacity array and give out the mass
    print("change in water vol",((numpy.sum(claw.frames[0].q[0,:]*(1/cells_number)*state.aux[1,:],axis=0))  - (numpy.sum(claw.frames[-1].q[0,:]*(1/cells_number)*state.aux[1,:],axis=0)))/numpy.sum(claw.frames[0].q[0,:]*(1/cells_number)*state.aux[1,:],axis=0))
    plot_kargs = {'problem_data':state.problem_data}
    plot(setplot="./setplot_h_box_wave.py",outdir=outdir,plotdir='./plots_diff_mom',iplot=False, htmlplot=True, **plot_kargs)

#setplot="./setplot_h_box_wave.py"

if __name__=="__main__":
#    from clawpack.pyclaw.util import run_app_from_main
#    setplot="./setplot_h_box_wave.py"
#    output = run_app_from_main(setup,setplot)
    setup()
