import BCPNN
import numpy as np
import TracePlotter
import pylab
import os
import sys

plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 16,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 1,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.10,
              'figure.subplot.bottom':.13,
              'figure.subplot.right':.92,
              'figure.subplot.top':.88,
              'figure.subplot.hspace':.30,
              'figure.figsize'  : TracePlotter.get_fig_size(800, portrait=False), 
              'figure.subplot.wspace':.25}


def compute_bcpnn_traces(spike_train_0, spike_train_1, K_vec, syn_params, t_sim, plot=False, extra_txt='', wij_lim=None):
    #######################
    # OFFLINE COMPUTATION
    #######################
    s_pre = BCPNN.convert_spiketrain_to_trace(spike_train_0, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(spike_train_1, t_sim)
    bcpnn_traces = []
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, syn_params, K_vec=K_vec)
    w_end = wij[-1]
    bcpnn_traces = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    if plot:
        dt = 0.1
        info_txt = 'nspikes pre %d\nnspikes post %d\nw_final: %.2f' % (len(spike_train_0), len(spike_train_1), w_end)
        TP = TracePlotter.TracePlotter(plot_params=plot_params)

        # either plot the K_vec
        TP.plot_zij_pij_weight_bias(bcpnn_traces, syn_params, dt, output_fn=None, fig=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', \
                extra_txt=extra_txt)
#                extra_txt=extra_txt, wij_lim=wij_lim)
    return w_end



if __name__ == '__main__':

    t_sim = 600.
    dt = 0.1
    t_shift = 100
#    st_0 = np.array([50., 55., 60., 65.])
    st_0 = np.arange(100, 200, 50)
    st_1 = st_0 + t_shift
    extra_txt = 'Two cells spike with $\\delta t = %d$\n' % (t_shift)
#    st_1 = np.array([150., 155., 160., 165.])
#    K_vec = np.ones((t_sim + 1.) / dt)
    spike_width = 0.1
    K_vec = np.ones((t_sim + spike_width) / dt)
    tau_p = 1000.
    tau_e = 1.
    tau_i = float(sys.argv[1])
#    tau_i = 20.
    tau_j = 5.
    weight_limits = (-4, 3)
    syn_params = {'p_i': .01, 'p_j': .01, 'p_ij': 1e-8, 'gain': 1.0, \
            'K': 0., 'fmax': 20., 'epsilon': 1. / (20 * tau_p), \
            'delay':1.0, 'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, syn_params, t_sim, plot=True, extra_txt=extra_txt, wij_lim=weight_limits)

#    output_fn = 'Figures_taup5000/example_plot_zij_pij_taui%04d.png' % tau_i
    output_fn = 'Figures/example_plot_zij_pij_taui%04d_tau_j%04d.png' % (tau_i, tau_j)
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=100)
    pylab.show()
