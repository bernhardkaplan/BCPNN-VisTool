import pylab
import numpy as np

def get_fig_size(fig_width_pt, portrait=False):
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    if portrait:
        fig_size = [fig_height,fig_width]
    else:
        fig_size =  [fig_width,fig_height]
    return fig_size

plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 14,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 1,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.13,
              'figure.subplot.right':.94,
              'figure.subplot.top':.92,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}
#              'figure.figsize': get_fig_size(800)}

plot_params_portrait = {'backend': 'png',
              'axes.labelsize': 24,
              'axes.titlesize': 24,
              'text.fontsize': 20,
              'xtick.labelsize': 24,
              'ytick.labelsize': 24,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 16,
               'lines.markersize': 1,
               'lines.linewidth': 3,
              'font.size': 12,
              'path.simplify': False,
              'figure.figsize': get_fig_size(800, portrait=True)}
#                  'figure.subplot.hspace':.25,}

class FigureCreator(object):

    def __init__(self):
        """
        Here, the standard parameters for a figure are set.
        """
        fig_width_pt = 800.0  
        pylab.rcParams.update(plot_params)


    def create_fig(self):
#        print "plotting ...."
#        self.n_fig_x = 1
#        self.n_fig_y = 1
        self.fig = pylab.figure(figsize=self.fig_size)
        return self.fig


    
    def create_fig_2d(self, data_array_2d, output_fn='', xlabel='', ylabel='', title=''):
        """
        Plots a standard 2-dimensional figure and returns the figure 
        Returns the 
        """





