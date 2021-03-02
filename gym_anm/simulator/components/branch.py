import numpy as np

from .errors import BranchSpecError
from .constants import BRANCH_H


class TransmissionLine(object):
    """
    A transmission line of an electric power grid.

    Attributes
    ----------
        f_bus : int
            The sending end bus ID :math:`i`.
        t_bus : int
            The receiving end bus ID :math:`j`.
        r : float
            The transmission line resistance :math:`r_{ij}` (p.u.).
        x : float
            The transmission line reactance :math:`x_{ij}` (p.u.).
        b : float
            The transmission line susceptance :math:`b_{ij}` (p.u.).
        rate : float
            The rate of the line :math:`\\overline S_{ij}` (p.u.).
        tap_magn : float
            The magnitude of the transformer tap :math:`\\tau_{ij}`.
        shift : float
            The complex phase angle of the transformer :math:`\\theta_{ij}` (radians).
        i_from, i_to : complex
            The complex current flows :math:`I_{ij}` and :math:`I_{ji}` (p.u.).
        p_from, p_to : float
            The real power flows :math:`P_{ij}` and :math:`P_{ji}` in the line (p.u.).
        q_from, q_to : float
            The reactive power flows :math:`Q_{ij}` and :math:`Q_{ji}` in the line (p.u.).
        s_apparent_max : float
            The apparent power flow through the line, taken as the maximum of the
            apparent power injection at each end, with the sign indicating its
            direction (+ if :math:`P_{ij} \\ge 0`; - otherwise) (p.u.).
        series, shunt : complex
            The series :math:`y_{ij}` and shunt :math:`y_{ij}^{sh}` admittances of the line in the pi-model (p.u.).
        tap : complex
            The complex tap of the transformer :math:`t_{ij}` (p.u.).
    """

    def __init__(self, br_spec, baseMVA, bus_ids):
        """
        Parameters
        ----------
        br_spec : numpy.ndarray
            The corresponding branch row in the network file describing the
            network.
        baseMVA : int
            The base power of the system (MVA).
        bus_ids : list of int
            The list of unique bus IDs.
        """

        self._check_input_specs(br_spec, baseMVA, bus_ids)
        self._compute_admittances()

        # Initialize attributes used later.
        self.i_from = None
        self.p_from = None
        self.q_from = None
        self.i_to = None
        self.p_to = None
        self.q_to = None
        self.s_apparent_max = None

    def _check_input_specs(self, br_spec, baseMVA, bus_ids):

        self.f_bus = br_spec[BRANCH_H['F_BUS']]
        if self.f_bus is None or self.f_bus not in bus_ids:
            raise BranchSpecError('The F_BUS value of the branch is {} but should be in {}.'.format(self.f_bus, bus_ids))
        else:
            self.f_bus = int(self.f_bus)

        self.t_bus = br_spec[BRANCH_H['T_BUS']]
        if self.t_bus is None or self.t_bus not in bus_ids:
            raise BranchSpecError('The T_BUS value of the branch is {} but should be in {}.'.format(self.t_bus, bus_ids))
        else:
            self.t_bus = int(self.t_bus)

        self.r = br_spec[BRANCH_H['BR_R']]
        if self.r is None:
            self.r = 0.
        elif self.r < 0:
            raise BranchSpecError('The BR_R value for branch (%d, %d) should be >= 0.' % (self.f_bus, self.t_bus))

        self.x = br_spec[BRANCH_H['BR_X']]
        if self.x is None:
            self.x = 0.
        elif self.x < 0:
            raise BranchSpecError('The BR_X value for branch (%d, %d) should be >= 0.' % (self.f_bus, self.t_bus))

        if self.r == 0 and self.x == 0:
            raise BranchSpecError('Branch (%d, %d) has r=x=0. This is not supported, as it will lead to infinite impedance.'
                                  'Possible workaround: set a small reactance x=0.0001.' % (self.f_bus, self.t_bus))

        self.b = br_spec[BRANCH_H['BR_B']]
        if self.b is None:
            self.b = 0.
        elif self.b < 0:
            raise BranchSpecError('The BR_B value for branch (%d, %d) should be >= 0.' % (self.f_bus, self.t_bus))

        self.rate = br_spec[BRANCH_H['RATE']]
        if self.rate is None:
            self.rate = np.inf
        elif self.rate < 0:
            raise BranchSpecError('The RATE value for branch (%d, %d) should be >= 0.' % (self.f_bus, self.t_bus))
        else:
            self.rate /= baseMVA

        self.tap_magn = br_spec[BRANCH_H['TAP']]
        if self.tap_magn is None:
            self.tap_magn = 1.
        elif self.tap_magn <= 0:
            raise BranchSpecError('The TAP value for branch (%d, %d) should be > 0. Use TAP=1 and SHIFT=0 to model'
                                  'the absence of an off-nominal transformer.' % (self.f_bus, self.t_bus))

        self.shift = br_spec[BRANCH_H['SHIFT']]
        if self.shift is None:
            self.shift = 0.
        elif self.shift < 0 or self.shift > 360:
            raise BranchSpecError('The BR_SHIFT value for branch (%d, %d) should be in [0, 360].' % (self.f_bus, self.t_bus))
        else:
            self.shift = self.shift * np.pi / 180

    def _compute_admittances(self):
        """
        Compute the series, shunt admittances and transformer tap of the line.
        """

        # Compute the branch series admittance as y_{ij} = 1 / (r + jx).
        self.series = 1. / (self.r + 1.j * self.x)

        # Compute the branch shunt admittance y_{ij}^{sh} = jb / 2.
        self.shunt = 1.j * self.b / 2.

        # Create complex tap ratio of generator as: tap = a exp(j shift).
        self.tap = self.tap_magn * np.exp(1.j * self.shift)

    def compute_currents(self, v_f, v_t):
        """
        Compute the complex current injections on the transmission line.

        Parameters
        ----------
        v_f : np.complex
            The complex voltage at bus :code:`self.f_bus`.
        v_t : np.complex
            The complex voltage at bus :code:`self.t_bus`.
        """

        # Forward current.
        i_1 = (self.series + self.shunt) * v_f / (np.absolute(self.tap) ** 2)
        i_2 = - self.series * v_t / np.conjugate(self.tap)
        self.i_from = i_1 + i_2

        # Backward current.
        i_1 = (self.series + self.shunt) * v_t
        i_2 = - self.series * v_f / self.tap
        self.i_to = i_1 + i_2

    def compute_power_flows(self, v_f, v_t):
        """
        Compute the power flows on the transmission line.

        Parameters
        ----------
        v_f : np.complex
            The complex voltage at bus :code:`self.f_bus` (p.u.).
        v_t : np.complex
            The complex voltage at bus :code:`self.t_bus` (p.u.).
        """

        # Forward power flows.
        s_from = v_f * np.conj(self.i_from)
        self.p_from = s_from.real
        self.q_from = s_from.imag

        # Backward power flows.
        s_to = v_t * np.conj(self.i_to)
        self.p_to = s_to.real
        self.q_to = s_to.imag

        # Compute directed apparent power flow.
        self.s_apparent_max = np.sign(self.p_from) \
                              * np.maximum(np.abs(s_from), np.abs(s_to))
