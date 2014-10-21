BCPNN-VisTool
=============

Just a simple tool to play with spiking BCPNN parameters.

The main file is test_functions.py where you can set the parameters for BCPNN and also build more advanced test scripts.
In test_functions.py is one example for a test script run_test_3 which does the following:
    init_K
    change_K
    simulate
    change_K
    simulate
For that you need to defien in boolean values the activity on the pre- and post synaptic side (and the K values for the corresponding changes).


If you have any questions, ask me.

Author: Bernhard Kaplan

Requirements: nest, BCPNN-module by Phil Tully
