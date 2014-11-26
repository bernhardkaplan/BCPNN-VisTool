import BCPNN
import numpy as np
import TracePlotter
import pylab

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
        TP = TracePlotter.TracePlotter()

        # either plot the K_vec
        TP.plot_trace_with_spikes(bcpnn_traces, syn_params, dt, output_fn=None, fig=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=K_vec, \
                extra_txt='K_vec')
        # or add some extra info_text in one of the subplots:
#        TP.plot_trace(bcpnn_traces, syn_params, dt, output_fn=None, info_txt=info_txt, fig=None, \
#                color_pre='b', color_post='g', color_joint='r', style_joint='-')
    return w_end


if __name__ == '__main__':

    t_sim = 500.
    dt = 0.1
    st_0 = np.array([50., 55., 60.])
    st_1 = np.array([150., 155., 160.])
#    K_vec = np.ones((t_sim + 1.) / dt)
    spike_width = 0.1
    K_vec = np.ones((t_sim + spike_width) / dt)
    tau_p = 1000.
    tau_e = 10.
    tau_i = 75.
    tau_j = 10.
    syn_params = {'p_i': .01, 'p_j': .01, 'p_ij': .0001, 'gain': 1.0, \
            'K': 0., 'fmax': 20., 'epsilon': 1. / (20. * tau_p), \
            'delay':1.0, 'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, syn_params, t_sim, plot=True)

    output_fn = 'example_plot_taui%d.png' % tau_i
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    pylab.show()
