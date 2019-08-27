from gym_smartgrid.constants import BUS_H


class Bus(object):
    """
    A bus (or node) of an electric power grid.

    Attributes
    ----------
    id : int
        The bus unique ID.
    type : int
        The bus type (1 = PQ, 2 = PV, 3 = slack).
    is_slack : bool
        True if it is the slack bus, False otherwise.
    v_min, v_max : float
        The minimum and maximum voltage magnitude (p_from.u.).
    v_slack : float
        The fixed voltage magnitude, if it is the slack bus (p_from.u.).
    v : float
        The current complex bus voltage (p_from.u.).
    p, q : float
        The current real (MW) and reactive (MVAr) power injections at the bus.
    p_min, p_max, q_min, q_max : float
        The bounds on the feasible real and reactive power injections at the bus.
    """

    def __init__(self, bus_case):
        """
        Parameters
        ----------
        bus_case : array_like
            The corresponding bus row in the case file describing the network.
        is_slak : bool, optional
            True the bus is the slack bus, False otherwise.
        """

        self.id = int(bus_case[BUS_H['BUS_I']])
        self.type = int(bus_case[BUS_H['BUS_TYPE']])
        self.baseKV = bus_case[BUS_H['BASE_KV']]
        self.v_max = bus_case[BUS_H['VMAX']]
        self.v_min = bus_case[BUS_H['VMIN']]

        if self.type == 3:
            self.is_slack = True
            self._v_slack = self.v_max
        else:
            self.is_slack = False

        self.v = None
        self.p = None
        self.q = None
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
            raise AttributeError(f'The bus with ID {self.id} is not the slack '
                                 f'bus.')

    @v_slack.setter
    def v_slack(self, value):
        self._v_slack = value