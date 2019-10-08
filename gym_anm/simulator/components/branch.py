import numpy as np

from gym_anm.constants import BRANCH_H


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
        rate : float
            The rate of the line in MVA.
        tap_magn : float
            The magnitude of the transformer tap.
        shift : float
            The complex phase angle of the transformer (degrees).
        i_from : complex
            The complex current flow in the line (p.u.).
        p_from, q_from : float
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
        br_case : numpy.ndarray
            The corresponding branch row in the network file describing the
            network.
        baseMVA : int
            The base power of the system (MVA).
        """

        # Import values from case file.
        self.f_bus = int(br_case[BRANCH_H['F_BUS']])
        self.t_bus = int(br_case[BRANCH_H['T_BUS']])
        self.r = br_case[BRANCH_H['BR_R']]
        self.x = br_case[BRANCH_H['BR_X']]
        self.b = br_case[BRANCH_H['BR_B']]
        self.rate = br_case[BRANCH_H['RATE']] / baseMVA
        self.tap_magn = br_case[BRANCH_H['TAP']]
        self.shift = br_case[BRANCH_H['SHIFT']]

        # Deal with unspecified values.
        self.tap_magn = self.tap_magn if self.tap_magn > 0. else 1.

        self._compute_admittances()

        # Initialize attributes used later.
        self.i_from = None
        self.p_from = None
        self.q_from = None
        self.i_to = None
        self.p_to = None
        self.q_to = None

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
