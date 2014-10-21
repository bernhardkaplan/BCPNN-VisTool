import TracePlotter
import numpy as np 
import pylab
import nest
import BCPNN


class TestBcpnn(object):

    def __init__(self):
        pass

    def init_sim(self, K):
        if 'bcpnn_synapse' not in nest.Models():
            try:
            #    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
                nest.Install('pt_module')
            except:
                nest.Install('pt_module')

        self.K_values = [K]
        self.times_K_changed = [0.]
        nest.SetKernelStatus({'overwrite_files': True, \
                'resolution' : 0.1})
        cell_params = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
            'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -80.0, 'V_th': -60.0, \
            'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 5.0, 'tau_syn_in': 5.0}
        self.nrns = nest.Create('iaf_cond_exp_bias', 2, params=cell_params)
        self.voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
        nest.SetStatus(self.voltmeter, [{"to_file": True, "withtime": True, 'label' : 'volt'}])
        nest.ConvergentConnect(self.voltmeter, self.nrns)
        self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'spikes'})
        nest.ConvergentConnect(self.nrns, self.exc_spike_recorder, delay=0.1, weight=1.)
        self.source_0 = nest.Create('spike_generator', 1)
        self.source_1 = nest.Create('spike_generator', 1)
        weight = 15.
        nest.Connect(self.source_0, [self.nrns[0]], weight, 1., model='static_synapse')
        nest.Connect(self.source_1, [self.nrns[1]], weight, 1., model='static_synapse')

        # neuron 0 -> neuron 1
        tau_i = 10.
        tau_j = 10.
        tau_e = 100.
        tau_p = 10000.
        fmax = 20.
        epsilon = 1. / (fmax * tau_p)
        bcpnn_init = 0.01
        p_i = bcpnn_init
        p_j = bcpnn_init
        weight = 0.
        p_ij = np.exp(weight) * p_i * p_j
        self.syn_params  = {'p_i': p_i, 'p_j': p_j, 'p_ij': p_ij, 'gain': 1.0, 'K': K, \
                'fmax': fmax, 'epsilon': epsilon, 'delay':1.0, \
                'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
        print 'syn_params init:', self.syn_params
        nest.Connect([self.nrns[0]], [self.nrns[1]], self.syn_params, model='bcpnn_synapse')
#        nest.SetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]), {'z_i': 

        self.t_sim = 0.


    def set_spiking_behavior(self, pre_spikes=True, post_spikes=True, t_start=0., t_stop=100., dt=1., n_rnd_spikes=5):

        # pre synaptic spikes
        if pre_spikes:
            spike_times_0 = np.round((t_stop - t_start) * np.random.random(n_rnd_spikes) + t_start, decimals=1) # spike_times_0 = np.arange(t_start, t_stop, n_rnd_spikes)
            spike_times_0 = np.sort(spike_times_0)
        else:
            spike_times_0 = np.array([])
        if post_spikes: 
            spike_times_1 = np.round((t_stop - t_start) * np.random.random(n_rnd_spikes) + t_start, decimals=1)
            spike_times_1 = np.sort(spike_times_1)
        else:
            spike_times_1 = np.array([])
        # post synaptic spikes 
        nest.SetStatus(self.source_0, {'spike_times' : spike_times_0})
        nest.SetStatus(self.source_1, {'spike_times' : spike_times_1})


    def get_nest_weight(self):
        conns = nest.GetConnections([self.nrns[0]], [self.nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
        if len(conns) > 0:
            cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
            pi = cp[0]['p_i']
            pj = cp[0]['p_j']
            pij = cp[0]['p_ij']
            w = np.log(pij / (pi * pj))
        return w

    def trigger_pre_spike_and_get_nest_weight(self):
        spike_times_0 = [self.t_sim + .1]
        nest.SetStatus(self.source_0, {'spike_times' : spike_times_0})
        nest.Simulate(7.0)
        self.t_sim += 7.0

        conns = nest.GetConnections([self.nrns[0]], [self.nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
        if len(conns) > 0:
            cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
            pi = cp[0]['p_i']
            pj = cp[0]['p_j']
            pij = cp[0]['p_ij']
            w = np.log(pij / (pi * pj))
            print 'weight after simulation:', w
        return w


    def get_spike_trains(self):
        """
        Returns the spike trains for the two neurons
        """
        spike_data = nest.GetStatus(self.exc_spike_recorder)[0]['events']
        gids = spike_data['senders']
        idx_0 = (gids == 1).nonzero()[0]
        st_0 = spike_data['times'][idx_0]
        idx_1 = (gids == 2).nonzero()[0]
        st_1 = spike_data['times'][idx_1]
        return st_0, st_1

    def simulate(self, t_sim):
        nest.Simulate(t_sim)
        self.t_sim += t_sim

    def set_kappa(self, K, gain=0.):
        self.K_values.append(K)
        self.times_K_changed.append(self.t_sim)
        nest.SetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]), {'K': K, 'gain': gain, 't_k': nest.GetKernelStatus()['time']})


    def get_K_vec(self, dt=0.1):
        K_vec = self.K_values[0] * np.ones(self.t_sim / dt + 10) # + 10 because the spikewidth = 10
        if len(self.times_K_changed) > 1:
            for i_, t_change in enumerate(self.times_K_changed[1:]):
                idx = int(t_change / dt)
                K_vec[idx:] = self.K_values[i_ + 1]
        return K_vec




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


def plot_voltages(volt_recorder):
    volt_data = nest.GetStatus(volt_recorder)[0]['events']
    gids = volt_data['senders']
    idx_0 = (gids == 1).nonzero()[0]
    idx_1 = (gids == 2).nonzero()[0]
    volt_0 = volt_data['V_m'][idx_0]
    volt_1 = volt_data['V_m'][idx_1]
    times_0 = volt_data['times'][idx_0]
    times_1 = volt_data['times'][idx_1]

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(times_0, volt_0, label='pre neuron')
    ax.plot(times_1, volt_1, label='post neuron')

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    pylab.legend()


def run_test_3(spike_pattern, K_values):
    """
    TEST 3:

        init_K
        change_K
        sim
        change_K
        sim

    """
    debug_txt = 'DEEEEEEEEEEEEEEEEEEBUG  spike_pattern %s \tK_values %s' % (str(spike_pattern), str(K_values))
    print debug_txt

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = 100
    Tester = TestBcpnn()
    ##### INIT K 
    Tester.init_sim(K=K_values[0])
    ##### CHANGE K
    Tester.set_kappa(K_values[1], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10.)
    ##### SIM
    Tester.simulate(t_sim)
    ##### CHANGE K AGAIN
    Tester.set_kappa(K_values[2], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[2], spike_pattern[3], t_start = 10 + t_sim, t_stop= 2 * t_sim-10.)
    ##### SIM
    Tester.simulate(t_sim)

    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    return w_diff_relative


if __name__ == '__main__':



    # This is a simple example with pre and post spiking
    spike_pattern = [True, True]    
    # spike_patterns are structured as [pre_spiking_in_period_0, post_spiking_in_period_0, ... other 'periods' ]
    # for more advances settings look at the function run_test_3 above (you have to increase the spike pattern according to your wishes,
    # e.g. pre_spiking, post_spiking, no_pre_spiking, post_spiking would be: 
    # e.g. run_test_3([True, True, False, True], [1., 1., 1.]) the second list being the K_values at different times of the simulation

    K_values = [1.] # 
    nest.ResetKernel()
    np.random.seed(0)
    t_sim = 1000.
    firing_rate = 15.
    n_spikes = firing_rate * t_sim / 1000.
    Tester = TestBcpnn()

    ##### INIT K 
    Tester.init_sim(K=K_values[0])
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)

    ##### SIM 
    Tester.simulate(t_sim)
    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    print 'DEBUG', st_0, st_1
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot=True)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    
    print 'w_nest:', w_nest
    print 'w_offline:', w_offline
    print 'w_diff_relative', w_diff_relative
    plot_voltages(Tester.voltmeter)
    pylab.show()
