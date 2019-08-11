from gym_smartgrid.constants import BRANCH_H


class TransmissionLine(object):
    """
    A transmission line of an electric power grid.

    Attributes
    ----------

    Parameters
    ----------
    br_case : array_like
        The corresponding branch row in the case file describing the network.

    Methods
    -------

    """


    def __init__(self, br_case):
        self.f_bus = int(br_case[BRANCH_H['F_BUS']])
        self.t_bus = int(br_case[BRANCH_H['T_BUS']])
        self.r = br_case[BRANCH_H['BR_R']]
        self.x = br_case[BRANCH_H['BR_X']]
        self.b = br_case[BRANCH_H['BR_B']]
        self.i_max = br_case[BRANCH_H['RATE_A']]
        self.tap = br_case[BRANCH_H['TAP']]
        self.shift = br_case[BRANCH_H['SHIFT']]
        self.ang_min = br_case[BRANCH_H['ANGMIN']]
        self.ang_max = br_case[BRANCH_H['ANGMAX']]

        self.tap = self.tap if self.tap > 0. else 1.

        self.i = None  # p.u. (complex)
        self.p = None  # MW
        self.q = None  # MVAr