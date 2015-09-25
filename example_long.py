import BCPNN
import numpy as np
import TracePlotter
import pylab

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
              'figure.subplot.hspace':.40,
              'figure.figsize'  : TracePlotter.get_fig_size(1000, portrait=False), 
              'figure.subplot.wspace':.25}


def compute_bcpnn_traces(spike_train_0, spike_train_1, K_vec, syn_params, t_sim, plot=False):
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
        TP.plot_trace_with_spikes(bcpnn_traces, syn_params, dt, output_fn=None, fig=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=K_vec, \
                extra_txt='Kappa value gates learning')
        # or add some extra info_text in one of the subplots:
#        TP.plot_trace(bcpnn_traces, syn_params, dt, output_fn=None, info_txt=info_txt, fig=None, \
#                color_pre='b', color_post='g', color_joint='r', style_joint='-')
    return w_end


if __name__ == '__main__':
    """
    This script computes the bcpnn traces for a pair of neurons in a setting with repetitive firing.
    """

    np.random.seed(0)
    dt = 0.1
    st_0 = []
    st_1 = []

    n_iterations = 5
    
    dt_stimulus_interval = 100.  # how far apart the stimulus packages are in time, dt_stimulus_interval = 0 would be very correlated (depending on f_max, and t_stim)
    t_stim = 100.               # length of one stimulus package
    dt_stim = 100.             # pause between the packages
    t_offset = 50.              # start of the first stimulus package
    f_max = 200.                 # f_max_stim, maximum response rate
    dt_events = 1. / f_max * t_stim
    n_events = f_max * t_stim / 1000.
    for i_ in xrange(n_iterations):
        st_0 += np.sort((i_ * dt_stim + t_stim + t_offset - i_ * dt_stim + t_offset) * np.random.random_sample(n_events) + i_ * dt_stim + t_offset).tolist()
        st_1 += np.sort((i_ * dt_stim + t_stim + t_offset + dt_stimulus_interval - i_ * dt_stim + t_offset + dt_stimulus_interval) * np.random.random_sample(n_events) + i_ * dt_stim + t_offset + dt_stimulus_interval).tolist()
        # for regular spike trains use:
#        st_0 += np.arange(i_ * dt_stim + t_offset, i_ * dt_stim + t_stim + t_offset, dt_events).tolist()

    st_0 = np.array(st_0)
    st_1 = np.array(st_1)
    t_sim = n_iterations * (t_stim  + dt_stim)
    print 'dt_events:', dt_events
    print 'n_events:', n_events
#    print 'st_0:', st_0
#    print 'st_1:', st_1

    spike_width = 0.1
#    K_vec = np.ones((t_sim + spike_width) / dt)
    K_vec = None
    tau_p = 10000.
    tau_e = 1.
    tau_i = 150.
    tau_j = 5.
    syn_params = {'p_i': .01, 'p_j': .01, 'p_ij': 1e-8, 'gain': 1.0, \
            'K': 0., 'fmax': 20., 'epsilon': 1. / (20 * tau_p), \
            'delay':1.0, 'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, syn_params, t_sim, plot=True)

    output_fn = 'training_niterations%d_taui%d_tstim%d_dtstim%d_fmaxstim%d.png' % (n_iterations, tau_i, t_stim, dt_stimulus_interval, f_max)
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    pylab.show()
