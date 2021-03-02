from .constants import BUS_H
from .errors import BusSpecError


class Bus(object):
    """
    A bus (or node) of an electric power grid.

    Attributes
    ----------
    id : int
        The bus unique ID :math:`i`.
    type : int
        The bus type (0 = slack, 1 = PQ).
    baseKV : float
        The base voltage of the region (kV).
    is_slack : bool
        True if it is the slack bus; False otherwise.
    v_min, v_max : float
        The minimum and maximum RMS voltage magnitudes :math:`[\\underline V_i,
        \\overline V_i]` (p.u.).
    v_slack : float
        The fixed voltage magnitude (used only for slack bus, set to :code:`v_max`) (p.u.).
    v : complex
        The complex bus voltage :math:`V_i` (p.u.).
    i : complex
        The complex bus current injection :math:`I_i` (p.u.).
    p, q : float
        The active :math:`P_i` (p.u.) and reactive :math:`Q_i` (p.u.) power injections at the bus.
    p_min, p_max, q_min, q_max : float
        The bounds on the feasible active and reactive power injections at the bus,
        computed as the sum of the bounds of the devices connected to the bus (p.u.).
    """

    def __init__(self, bus_spec):
        """
        Parameters
        ----------
        bus_spec : array_like
            The corresponding bus row in the network file describing the network.
        """

        self.id = int(bus_spec[BUS_H['BUS_ID']])
        self.type = int(bus_spec[BUS_H['BUS_TYPE']])
        self.baseKV = bus_spec[BUS_H['BASE_KV']]
        self.v_max = bus_spec[BUS_H['VMAX']]
        self.v_min = bus_spec[BUS_H['VMIN']]

        if self.type == 0:
            self.is_slack = True
            self._v_slack = self.v_max
        else:
            self.is_slack = False

        self._check_input_specs()

        self.v = None
        self.p = None
        self.q = None
        self.i = None
        self.p_min = None
        self.p_max = None
        self.q_min = None
        self.q_max = None

    @property
    def v_slack(self):
        """
        Raises
        ------
        AttributeError
            If the bus is not the slack bus.
        """
        if self.is_slack:
            return self._v_slack
        else:
            raise AttributeError('The bus with ID {} is not the slack '
                                 'bus.'.format(self.id))

    @v_slack.setter
    def v_slack(self, value):
        self._v_slack = value

    def _check_input_specs(self):

        if self.type is None or self.type not in [0, 1]:
            raise BusSpecError('The BUS_TYPE value for bus %d should be in [0, 1]' % (self.id))

        if self.baseKV is None or self.baseKV <= 0:
            raise BusSpecError('The BASE_KV value for bus %d should be > 0' % (self.id))

        if self.v_max is None or self.v_max < 0:
            raise BusSpecError('The VMAX value for bus %d should be >= 0' % (self.id))

        if self.v_min is None or self.v_min < 0:
            if not self.is_slack:
                raise BusSpecError('The VMIN value for bus %d should be >= 0' % (self.id))

        if not self.is_slack and self.v_max < self.v_min:
            raise BusSpecError('Bus %d has VMAX < VMIN' % (self.id))
