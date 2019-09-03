import numpy as np

from gym_smartgrid.constants import BUS_H, DEV_H


def check_casefile(case):

    buses = case['bus']
    gen = case['gen']

    # Check if there is exactly 1 slack bus, as specified in the case file.
    if np.sum(buses[:, BUS_H['BUS_TYPE']] == 3) != 1:
        raise ValueError('There should be exactly one slack bus, '
                         'as specified in the TYPE field of case["bus"].')

    # Check the correctness of the slack bus specifications and get its
    # fixed voltage magnitude.
    if buses[0, BUS_H['BUS_TYPE']] == 3:

        # Check devices that are given as PV variables (fixed P and |V|).
        # There should be exactly one such device (the slack bus) and it
        # should be the first bus in the input file list. An error is
        # raised otherwise.

        # Check if there is exactly one PV generator.
        if np.sum(gen[:, DEV_H['VG']] != 0.) != 1:
            raise ValueError('There should be exactly 1 PV generator '
                             'connected to the slack (first) bus.')

        # Check if the PV generator is the first one in the list of
        # generators specified.
        if gen[0, DEV_H['VG']] == 0.:
            raise ValueError('The first generator in the input file should '
                             'be a PV generator, connected to the slack '
                             'bus.')
        # Check if there is exactly 1 slack device specified in the
        # VRE_TYPE column and it is the first one.
        if np.sum(gen[:, DEV_H['VRE_TYPE']] == 0.) != 1 \
                or gen[0, DEV_H['VRE_TYPE']] != 0.:
            raise ValueError('The first device in the case.gen table '
                             'should have VRE_TYPE == 0. to signify slack '
                             'bus, and no other device should.')

    else:
        raise ValueError("The slack bus of the test case must be specified "
                         "as the first bus in the input file. case['bus']["
                         "0, 1] == 3 should be true.")

    # Check that the set-point voltage magnitude at the slack bus
    # seems coherent.
    if (self.V_magn_slack < 0.5) or (self.V_magn_slack > 1.5):
        warnings.warn("Warning: voltage magnitude (" + str(self.V_magn_slack)
                      + ") at the slack bus does not seem coherent.")