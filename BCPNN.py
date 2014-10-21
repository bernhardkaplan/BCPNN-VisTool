import numpy as np

def convert_spiketrain_to_trace(st, t_max, dt=0.1, spike_width=10):
    """Converts a single spike train into a binary trace
    Keyword arguments: 
    st --  spike train in the format [time, id]
    spike_width -- number of time steps (in dt) for which the trace is set to 1
    Returns a np.array with st[i] = 1 if i in st[:, 0], st[i] = 0 else.

    TODO: get t_min
    """
    n = np.int(t_max / dt) + spike_width
    trace = np.zeros(n)
    spike_idx = st / dt
    idx = spike_idx.astype(np.int)
    trace[idx] = 1
    for i in xrange(spike_width):
        trace[idx + i] = 1
    return trace


def get_spiking_weight_and_bias(pre_trace, post_trace, bcpnn_params, dt=.1, K_vec=None, w_init=0.):
    """
    Arguments:
        pre_trace, post_trace: pre-synaptic activity (0 means no spike, 1 means spike) (not spike trains!)
        bcpnn_params: dictionary containing all bcpnn parameters, initial value, fmax, time constants, etc
        dt -- should be the simulation time step because it influences spike_height
    """
    assert (len(pre_trace) == len(post_trace)), "Bcpnn.get_spiking_weight_and_bias: pre and post activity have different lengths!"
    if K_vec != None:
        assert (len(K_vec) == len(pre_trace)), "Bcpnn.get_spiking_weight_and_bias: pre-trace and Kappa-Vector have different lengths!\nlen pre_trace %d K_vec %d" % \
                (len(pre_trace), len(K_vec))

    initial_value = bcpnn_params['p_i']
    n = len(pre_trace)
    si = pre_trace      # spiking activity (spikes have a width and a height)
    sj = post_trace

    zi = np.ones(n) * 0.01
    zj = np.ones(n) * 0.01
    eij = np.ones(n) * 0.001
    ei = np.ones(n) * 0.01
    ej = np.ones(n) * 0.01
    pi = np.ones(n) * 0.01
    pj = np.ones(n) * 0.01

#    zi = np.ones(n) * initial_value
#    zj = np.ones(n) * initial_value
#    eij = np.ones(n) * initial_value**2
#    ei = np.ones(n) * initial_value
#    ej = np.ones(n) * initial_value
#    pi = np.ones(n) * initial_value
#    pj = np.ones(n) * initial_value
    pij = pi * pj * np.exp(w_init)
    wij = np.ones(n)  * w_init #np.log(pij[0] / (pi[0] * pj[0]))
    bias = np.ones(n) * np.log(initial_value)
#    spike_height = 1000. / (bcpnn_params['fmax'] * dt)
    spike_height = 1000. / bcpnn_params['fmax']
    eps = bcpnn_params['epsilon']
    K = bcpnn_params['K']
    gain = bcpnn_params['gain']
    if K_vec == None:
        K_vec = np.ones(n) * K

    for i in xrange(1, n):
#        print 'debug', K_vec[i]
        # pre-synaptic trace zi follows si
        dzi = dt * (si[i] * spike_height - zi[i-1] + eps) / bcpnn_params['tau_i']
        zi[i] = zi[i-1] + dzi

        # post-synaptic trace zj follows sj
        dzj = dt * (sj[i] * spike_height - zj[i-1] + eps) / bcpnn_params['tau_j']
        zj[i] = zj[i-1] + dzj

        # pre-synaptic trace zi follows zi
        dei = dt * (zi[i] - ei[i-1]) / bcpnn_params['tau_e']
        ei[i] = ei[i-1] + dei

        # post-synaptic trace ej follows zj
        dej = dt * (zj[i] - ej[i-1]) / bcpnn_params['tau_e']
        ej[i] = ej[i-1] + dej

        # joint eij follows zi * zj
        deij = dt * (zi[i] * zj[i] - eij[i-1]) / bcpnn_params['tau_e']
        eij[i] = eij[i-1] + deij

        # pre-synaptic probability pi follows zi
        dpi = dt * K_vec[i] * (ei[i] - pi[i-1]) / bcpnn_params['tau_p']
        pi[i] = pi[i-1] + dpi

        # post-synaptic probability pj follows ej
        dpj = dt * K_vec[i] * (ej[i] - pj[i-1]) / bcpnn_params['tau_p']
        pj[i] = pj[i-1] + dpj

        # joint probability pij follows e_ij
        dpij = dt * K_vec[i] * (eij[i] - pij[i-1]) / bcpnn_params['tau_p']
        pij[i] = pij[i-1] + dpij

    # weights
    wij = gain * np.log(pij / (pi * pj))

    # bias
    bias = gain * np.log(pj)

    return [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj]



