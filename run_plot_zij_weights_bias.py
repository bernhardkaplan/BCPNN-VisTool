import os
import numpy as np

#tau_zis = [5]
#tau_zis = np.arange(5, 100, 5)
#tau_zis = np.r_[tau_zis, np.array([150, 200, 250, 300, 400, 500])]
tau_zis = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]
for tau_zi in tau_zis:
    os.system('python plot_zij_pij_weights_bias.py %d' % (tau_zi))


