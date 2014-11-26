import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import sys
import os
import numpy as np
import pylab

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


class TracePlotter(object):

    def __init__(self):

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
                      'figure.subplot.wspace':.18}

        pylab.rcParams.update(plot_params)


    def plot_trace_with_spikes(self, bcpnn_traces, bcpnn_params, dt, output_fn=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
            extra_txt=None):
        # unpack the bcpnn_traces
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, pre_trace, post_trace = bcpnn_traces
        t_axis = dt * np.arange(zi.size)
        plots = []
#        pylab.rcParams.update({'figure.subplot.hspace': 0.22, '})
        if fig == None:
            fig = pylab.figure(figsize=get_fig_size(1200, portrait=False))
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
        else:
            ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()
        linewidth = 3
        legend_fontsize=20
        
        self.title_fontsize = 24
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (bcpnn_params['tau_i'], bcpnn_params['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c=color_pre, lw=linewidth, ls=':')
        ax1.plot(t_axis, post_trace, c=color_post, lw=linewidth, ls=':')
        p1, = ax1.plot(t_axis, zi, c=color_pre, label='$z_i$', lw=linewidth)
        p2, = ax1.plot(t_axis, zj, c=color_post, label='$z_j$', lw=linewidth)
        plots += [p1, p2]
        labels_z = ['Pre $z_i$', 'Post $z_j$']
        ax1.legend(plots, labels_z, fontsize=legend_fontsize, loc='lower right')
#        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax5.plot(t_axis, pi, c=color_pre, lw=linewidth)
        p2, = ax5.plot(t_axis, pj, c=color_post, lw=linewidth)
        p3, = ax5.plot(t_axis, pij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax5.set_title('$\\tau_{p} = %d$ ms' % \
                (bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax5.legend(plots, labels_p, fontsize=legend_fontsize, loc='lower right')
        ax5.set_xlabel('Time [ms]')
        ax5.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c=color_pre, lw=linewidth)
        p2, = ax3.plot(t_axis, ej, c=color_post, lw=linewidth)
        p3, = ax3.plot(t_axis, eij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (bcpnn_params['tau_e']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p, fontsize=legend_fontsize, loc='lower right')
#        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
#        labels_w = ['$w_{ij}$']
        labels_w = ['$w_{ij} = gain \cdot log(\\frac{p_{ij}}{p_i \cdot p_j})$']
#        wij = gain * np.log(pij / (pi * pj))
        ax4.legend(plots, labels_w, fontsize=legend_fontsize, loc='lower right')
#        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_, fontsize=legend_fontsize, loc='lower right')
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

        if K_vec != None:
            p1, = ax2.plot(t_axis, K_vec, c='k', lw=linewidth)
            ax2.set_ylabel('Kappa')
        if extra_txt != None:
            ax2.set_title(extra_txt)

        if output_fn != None:
            print 'Saving traces to:', output_fn
            pylab.savefig(output_fn)

        return fig



    def plot_trace(self, bcpnn_traces, bcpnn_params, dt, output_fn=None, info_txt=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-'):
        # unpack the bcpnn_traces
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, pre_trace, post_trace = bcpnn_traces
        t_axis = dt * np.arange(zi.size)
        plots = []
        pylab.rcParams.update({'figure.subplot.hspace': 0.50})
        if fig == None:
            fig = pylab.figure(figsize=get_fig_size(1200, portrait=False))
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
        else:
            ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()
        linewidth = 1
        self.title_fontsize = 24
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (bcpnn_params['tau_i'], bcpnn_params['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c=color_pre, lw=linewidth, ls=':')
        ax1.plot(t_axis, post_trace, c=color_post, lw=linewidth, ls=':')
        p1, = ax1.plot(t_axis, zi, c=color_pre, label='$z_i$', lw=linewidth)
        p2, = ax1.plot(t_axis, zj, c=color_post, label='$z_j$', lw=linewidth)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z, loc='lower right')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax2.plot(t_axis, pi, c=color_pre, lw=linewidth)
        p2, = ax2.plot(t_axis, pj, c=color_post, lw=linewidth)
        p3, = ax2.plot(t_axis, pij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax2.set_title('$\\tau_{p} = %d$ ms' % \
                (bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax2.legend(plots, labels_p, loc='lower right')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c=color_pre, lw=linewidth)
        p2, = ax3.plot(t_axis, ej, c=color_post, lw=linewidth)
        p3, = ax3.plot(t_axis, eij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (bcpnn_params['tau_e']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p, loc='lower right')
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w, loc='lower right')
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_, loc='lower right')
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

        ax5.set_yticks([])
        ax5.set_xticks([])

        w_max, w_end, w_avg = np.max(wij), wij[-1], np.mean(wij)

#        info_txt_ = 'Weight max: %.2e\nWeight end: %.2e\nWeight avg: %.2e\n' % \
#                (w_max, w_end, w_avg)
#        info_txt_ += info_txt
        if info_txt != '':
            ax5.annotate(info_txt, (.02, .05), fontsize=18)


#        ax5.set_xticks([])
#        output_fn = self.params['figures_folder'] + 'traces_tauzi_%04d_tauzj%04d_taue%d_taup%d_dx%.2e_dv%.2e_vstim%.1e.png' % \
#                (bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'], self.dx, self.dv, self.v_stim)
#        output_fn = self.params['figures_folder'] + 'traces_dx%.2e_dv%.2e_vstim%.1e_tauzi_%04d_tauzj%04d_taue%d_taup%d.png' % \
#                (self.dx, self.dv, self.v_stim, bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'])


        if output_fn != None:
            print 'Saving traces to:', output_fn
            pylab.savefig(output_fn)

        return fig


