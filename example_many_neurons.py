import BCPNN
import numpy as np
import TracePlotter
import pylab
import itertools

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


def compute_many_bcpnn_traces(spike_train_0, spike_train_1, K_vec, syn_params, t_sim, extra_txt=''):
    """
    TP -- trace plotter, should be None for the first call of this function, then the old plotter is used
            for plotting additional traces
    """
    s_pre = BCPNN.convert_spiketrain_to_trace(spike_train_0, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(spike_train_1, t_sim)
    bcpnn_traces = []
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, syn_params, K_vec=K_vec)
    w_end = wij[-1]
    bcpnn_traces = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    return bcpnn_traces



def generate_spiketrains(n_neurons, t_stim, dt_stim, t_pause, n_stim, f_pre, f_post, t_offset):
    """
    Returns a pair of lists containing the pre- and the post-spikes.
    n_neurons   --  number of neurons per pre- and per post-population
    t_stim      --  time of one stimulation (during which the pre- and post-population is active)
    dt_stim     --  time between pre and post activation
    t_pause     --  pause between one pre-post activation stimulation
    f_pre       --  firing rate of the pre-population
    f_post      --  firing rate of the post-population
    """


    pre_spiketrains = []
    n_events_pre = f_pre * t_stim / 1000.
    for i_nrn in xrange(n_neurons):
        st_0 = []
        for i_stim in xrange(n_stim):
            t0 = i_stim * t_stim + t_offset + np.round(i_stim - 1 + .51) * (dt_stim + t_pause) # round is (i_stim - 1 + .51) is to consider the -1 of pauses for t_offset
            t1 = t0 + t_stim # upper boundary
            st_0 += np.sort((t1 - t0) * np.random.random_sample(n_events_pre) + t0).tolist()
        pre_spiketrains.append(st_0)

    post_spiketrains = []
    n_events_post = f_post * t_stim / 1000.
    for i_nrn in xrange(n_neurons):
        st_0 = []
        for i_stim in xrange(n_stim):
            t0 = i_stim * t_stim + t_offset + np.round(i_stim - 1 + .51) * (dt_stim + t_pause) # round is (i_stim - 1 + .51) is to consider the -1 of pauses for t_offset
            t1 = t0 + t_stim # upper boundary
            st_0 += np.sort((t1 - t0) * np.random.random_sample(n_events_post) + t0).tolist()
        post_spiketrains.append(st_0)

    return (pre_spiketrains, post_spiketrains)



if __name__ == '__main__':
    """
    This script computes the bcpnn traces for a pair of neurons in a setting with repetitive firing.
    """

    n_neurons = 4
    n_stim = 50
    dt_stimulus_interval = 0.  # how far apart the stimulus packages are in time, dt_stimulus_interval = 0 would be very correlated (depending on f_max, and t_stim)
    t_stim = 100.               # length of one stimulus package
    t_offset = 50.              # start of the first stimulus package
    t_pause = 1000.             # pause between one pre-post activation stimulation
    f_pre = 20.                 # f_pre, maximum response rate of pre-synaptic population
    f_post = 20.                # f_post, rate of post-synaptic population
    dt = 0.1                    # for trace computation
    t_sim = n_stim * (t_stim  + dt_stimulus_interval + t_pause) + 2 * t_offset
    seed = 0
    np.random.seed(seed)
    (pre_spiketrains, post_spiketrains) = generate_spiketrains(n_neurons, t_stim, dt_stimulus_interval, t_pause, n_stim, f_pre, f_post, t_offset)

    np.random.seed(0)

#    spike_width = 0.1
#    K_vec = np.ones((t_sim + spike_width) / dt)
    K_vec = None
    tau_p = 1000.
    tau_e = .1
    tau_i = 150.
    tau_j = 150.
    fmax = 20.
    init_value = 0.001
    epsilon = 1. / (fmax * tau_p)
    syn_params = {'p_i': init_value, 'p_j': init_value, 'p_ij': init_value**2, 'gain': 1.0, \
            'K': 0., 'fmax': fmax, 'epsilon': epsilon, \
            'delay':1.0, 'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}

    extra_txt = '$t_{stim}=%d\\  \\Delta t_{stim}=%d\\  t_{pause}=%d$ [ms]\n' % (t_stim, dt_stimulus_interval, t_pause)
    TP = TracePlotter.TracePlotter(plot_params=plot_params)

    w_finals = []
    for (pre, post) in itertools.product(range(n_neurons), range(n_neurons)):
        st_0 = np.array(pre_spiketrains[pre])
        st_1 = np.array(post_spiketrains[post])
        bcpnn_traces = compute_many_bcpnn_traces(st_0, st_1, K_vec, syn_params, t_sim)
        [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post] = bcpnn_traces
        w_finals.append(wij[-1])
        wij_label = '$w_{ij}^{final} = %.2f \\pm %.2f$' % (np.mean(w_finals), np.std(w_finals))
        TP.plot_zij_pij_weight_bias(bcpnn_traces, syn_params, dt, output_fn=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', extra_txt=extra_txt, wij_label=wij_label)

    output_fn = 'training_nstim%d_taui%d_tauj%d_tstim%d_dtstim%d_tpause%d.png' % (n_stim, tau_i, tau_j, t_stim, dt_stimulus_interval, t_pause)
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    pylab.show()
