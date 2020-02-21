import numpy
import matplotlib.pyplot as plt
from clawpack import pyclaw
def load_parameters(fileName):
    fileObj = open(fileName)
    params = {}
    for line in fileObj:
        line = line.strip()
        key_value = line.split('=')
        params[key_value[0]] = key_value[1]
    return params

# fort_output1 = '/home/cr2940/SWE1D_barrier_hbox/_output_cellwall_small_16000/fort.q00'
# h1 = numpy.zeros((21,16000))
# hu1 = numpy.zeros((21,16000))
# hi = 0
# hui = 0
# for i in range(21):
#     # get the right time step output
#     num = str(i)
#     if i < 10:
#         fort_output_1 = fort_output1 + '0' + num
#     else:
#         fort_output_1 = fort_output1 + num
#     output1 = open(fort_output_1,'r')
#     # skip the first 5 lines to jump to the height and momentum values:
#     for k in range(6):
#         output1.readline()
#     # extract the h and hu values:
#     h = []
#     hu =[]
#     for line in output1:
#         #if even%2 == 0 and k != num_cells:
#         line = line.rstrip("\n")
#         hhu1 = line.split()
#         h.append(float(hhu1[0]))
#         hu.append(float(hhu1[1]))
# #    h.append(float(h[-1]))
# #    hu.append(float(hu[-1]))
#     h1[i,:] = numpy.asarray(h)
#     hu1[i,:] = numpy.asarray(hu)
# # print(h1[0,:],hu1[0,:])
# --------------------------
def setplot(plotdata,problem_data):
#--------------------------
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """
    plotdata.clearfigures()  # clear any old figures,axes,items data

    params = load_parameters('parameters_h_box_wave.txt')
    xlower = float(params['xlower'])
    xupper = float(params['xupper'])
    cells_number = int(params['cells_number'])
    regular_cells_number = cells_number - 1
    nw = int(params['wall_position'])
    alpha = float(params['fraction'])
    wall_height = float(params['wall_height'])
    delta_x = (xupper - xlower) / regular_cells_number
    delta_xc = (xupper - xlower) / cells_number

    nw_edge_1_c = (nw-1) * delta_xc + xlower
    nw_edge_c = nw * delta_xc + xlower
    nw_edge_2_c = (nw+1) * delta_xc + xlower

    nw_edge_1_p = (nw-1) * delta_x + xlower
    nw_edge_p = (nw-1 + alpha) * delta_x + xlower
    nw_edge_2_p = nw * delta_x + xlower



    def mapping_h_box(xc):
        xp = xc + 0.0
        ratio = cells_number * 1.0 / regular_cells_number # delta_xp / delta_xc
        idx = numpy.where((xc>=nw_edge_1_c)&(xc<=nw_edge_2_c))[0]

        xp[:idx[0]] = (xc[:idx[0]] - xlower) * ratio + xlower
        xp[idx[-1]+1:] = xupper - (xupper - xc[idx[-1]+1:]) * ratio
        for i in idx:
            if nw_edge_1_c < xc[i] <= nw_edge_c:
                xp[i] = nw_edge_1_p + (xc[i] - nw_edge_1_c) * alpha * ratio
            elif nw_edge_c < xc[i] < nw_edge_2_c:
                xp[i] = nw_edge_p + (xc[i] - nw_edge_c) * (1 - alpha) * ratio

        return xp



    plotdata.mapc2p = mapping_h_box

    # Plot variables
    def bathy(current_data):
        return  current_data.aux[0, :]



    # def height2(current_data):
    #     x = pyclaw.Dimension(xlower, xupper, cells_number, name='x')
    #     domain = pyclaw.Domain(x)
    #     state = pyclaw.State(domain, 2, 2)
    #
    #     xc = state.grid.x.centers
    #     print(len(xc))
    #     print(len(h1[current_data.frameno,:]))
    #     axis = plt.gca()
    #     jump = numpy.zeros(len(xc))
    #     jump[nw] = 0.1 #wall height
    #     axis.plot(xc,h1[current_data.frameno,:]+bathy(current_data)+jump,'r:')
    #
    #     x_wall = nw_edge_p
    #     y1 =  current_data.aux[0,nw-1]
    #     y2 =  y1 + wall_height
    #     axis.plot([x_wall,x_wall],[y1,y2],'g',linewidth=2.5)

    def eta(current_data):
        return current_data.q[0, :] + bathy(current_data)
    def height(current_data):
        return current_data.q[0, :]
    def momentum(current_data):
        return current_data.q[1, :]
    #
    # def momentum2(current_data):
    #     x = pyclaw.Dimension(xlower, xupper, cells_number, name='x')
    #     domain = pyclaw.Domain(x)
    #     state = pyclaw.State(domain, 2, 2)
    #
    #     xc = state.grid.x.centers
    #     x_wall = nw_edge_p
    #     axis = plt.gca()
    #     axis.plot(xc,hu1[current_data.frameno,:],'r:')
    #     axis.plot([x_wall,x_wall],[-1.1*current_data.q[1,nw],1.1*current_data.q[1,nw]],":")

    def cell_ref_lines(current_data):
        # plt.hold(True)
        x = numpy.linspace(xlower, xupper, regular_cells_number+2, endpoint=True)
        x_edge = numpy.zeros(cells_number+1)
        x_edge[:nw] = x[:nw]
        x_edge[-nw:] = x[-nw:]
        x_edge[nw] = x_edge[nw-1] + alpha*delta_x
        x_edge[nw+1] = x_edge[nw] + (1-alpha)*delta_x
        x_edge[nw+2:] = x[nw+2:]
        y_edge_1 =  -1.0
        y_edge_2 =  1.0
        axis = plt.gca()
        axis.plot([x_edge,x_edge],mass_ylimits,'b--',linewidth=0.5)
        x_wall = x_edge[nw]
        y1 =  current_data.aux[0,nw-1]
        y2 =  y1 + wall_height
        axis.plot([x_wall,x_wall],[y1,y2],'r',linewidth=2)

    def momentum_ref_lines(current_data):
        x = numpy.linspace(xlower, xupper, regular_cells_number+2, endpoint=True)
        x_edge = numpy.zeros(cells_number+1)
        x_edge[:nw] = x[:nw]
        x_edge[-nw:] = x[-nw:]
        x_edge[nw] = x_edge[nw-1] + alpha*delta_x
        y_edge_1 =  -1.0
        y_edge_2 =  1.0
        axis = plt.gca()
        axis.plot([x_edge,x_edge],momentum_ylimits,'b--',linewidth=0.5)
        axis.plot([xlower,xupper],[0.0,0.0],'r--',linewidth=0.5)

    def wall_ref_lines(current_data):
        x_wall = nw_edge_p
        y1 =  current_data.aux[0,nw-1]
        y2 =  y1 + wall_height
        axis = plt.gca()
        axis.plot([x_wall,x_wall],[y1,y2],'g',linewidth=2.5)

    rgb_converter = lambda triple: [float(rgb) / 255.0 for rgb in triple]
    mass_ylimits = [-1.2,0]
    x_limits = [xlower, xupper]
    momentum_ylimits = [-1.0, 1.0]

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Depth', figno=0)
    plotfigure.kwargs = {'figsize': [6,6.4]}

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = x_limits
    plotaxes.ylimits = mass_ylimits
    plotaxes.title = 'Water Depth'
    plotaxes.afteraxes = wall_ref_lines
    # plotaxes.afteraxes = height2
    plotaxes.axescmd = 'subplot(211)'
    plotType = '1d_pwconst'
    # plotType='1d_plot'
    # def phys_grid(current_data):
    #     from pylab import plot
    #
    #     xlower = current_data.xlower
    #     xupper = current_data.xupper
    #



    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    plotitem.plot_var = eta
    plotitem.plot_var2 = bathy
    plotitem.color = rgb_converter((0,0,255))
    plotitem.mapc2p = mapping_h_box
    #plotitem.afteritem = phys_grid

    def ground(current_data):
        return -1.2*numpy.ones(len(current_data.q[0,:]))
    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    # plotitem = plotaxes.new_plotitem(plot_type='1d_pwconst')
    plotitem.plot_var = bathy
    plotitem.plot_var2 = ground
    plotitem.color = rgb_converter((139,69,19))

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_pwconst')
    plotitem.plot_var = eta
    plotitem.color = 'k'
    #plotitem.plotstyle= 'o'


    # Axes for momentum
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = x_limits
    plotaxes.ylimits = momentum_ylimits
    plotaxes.title = 'Momentum'
    # plotaxes.afteraxes = momentum2

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_pwconst')
    plotitem.plot_var = momentum
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

## ZOOM IN ####
    # Figure for depth (zoom in)
    plotfigure = plotdata.new_plotfigure(name='Depth_zoom', figno=1)
    plotfigure.kwargs = {'figsize': [6,6.4]}

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [delta_x * (nw-5)+xlower, delta_x * (nw+5)+xlower]
    plotaxes.ylimits = mass_ylimits
    plotaxes.title = 'Water Depth'
    plotaxes.afteraxes = cell_ref_lines
    plotaxes.axescmd = 'subplot(211)'


    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = bathy
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eta
    plotitem.color = 'k'
    plotitem.plotstyle = '*'

    # Axes for momentum
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [delta_x * (nw-5)+xlower, delta_x * (nw+5)+xlower]
    plotaxes.ylimits = momentum_ylimits
    plotaxes.title = 'Momentum'
    plotaxes.afteraxes = momentum_ref_lines
    plotaxes.axescmd = 'subplot(212)'

    plotitem = plotaxes.new_plotitem(plot_type=plotType)
    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = momentum
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}


    # plotdata.print_format = 'pdf'            # file format

    return plotdata
