import numpy as np

from gym_smartgrid.constants import BRANCH_H


class TransmissionLine(object):
    """
    A transmission line of an electric power grid.

    Attributes
    ----------
        f_bus : int
            The sending end bus ID.
        t_bus : int
            The receiving end bus ID.
        r : float
            The transmission line resistance (p.u.).
        x : float
            The transmission line reactance (p.u.).
        b : float
            The transmission line susceptance (p.u.).
        i_max : float
            The current rate of the line (p.u.).
        tap_magn : float
            The magnitude of the transformer tap.
        shift : float
            The complex phase angle of the transformer (degrees).
        ang_min, arg_max : float
            The minimum and maximum angle phase shifts across the line (degrees).
        i : complex
            The complex current flow in the line (p.u.).
        p, q : float
            The real (MW) and reactive (MVAr) power flow in the line.
        series, shunt : complex
            The series and shunt admittances of the line in the pi-model (p.u.).
        tap : complex
            The complex tap of the transformer.
    """

    def __init__(self, br_case, baseMVA):
        """
        Parameters
        ----------
        br_case : array_like
            The corresponding branch row in the case file describing the network.
        baseMVA : int
            The base power of the system (MVA).
        """

        # Import values from case file.
        self.f_bus = int(br_case[BRANCH_H['F_BUS']])
        self.t_bus = int(br_case[BRANCH_H['T_BUS']])
        self.r = br_case[BRANCH_H['BR_R']]
        self.x = br_case[BRANCH_H['BR_X']]
        self.b = br_case[BRANCH_H['BR_B']]
        self.i_max = br_case[BRANCH_H['RATE_A']] / baseMVA
        self.tap_magn = br_case[BRANCH_H['TAP']]
        self.shift = br_case[BRANCH_H['SHIFT']]
        self.ang_min = br_case[BRANCH_H['ANGMIN']]
        self.ang_max = br_case[BRANCH_H['ANGMAX']]

        # Deal with unspecified values.
        self.tap_magn = self.tap_magn if self.tap_magn > 0. else 1.

        self._compute_admittances()

        # Initialize attributes used later.
        self.i = None
        self.p = None
        self.q = None

    def _compute_admittances(self):
        """
        Compute the series, shunt admittances and transformer tap of the line.
        """

        # Compute the branch series admittance as y_s = 1 / (r + jx).
        self.series = 1. / (self.r + 1.j * self.x)

        # Compute the branch shunt admittance y_m = jb / 2.
        self.shunt = 1.j * self.b / 2.

        # Create complex tap ratio of generator as: tap = a exp(j shift).
        shift = self.shift * np.pi / 180.
        self.tap = self.tap_magn * np.exp(1.j * shift)
