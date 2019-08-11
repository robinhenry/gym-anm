from gym_smartgrid.constants import BUS_H


class Bus(object):
    """
    A bus (or node) of an electric power grid.

    Attributes
    ----------

    Parameters
    ----------
    bus_case : array_like
        The corresponding bus row in the case file describing the network.
    is_slak : bool, optional
        Indicate whether the bus is the slack bus.

    Methods
    -------

    """
    def __init__(self, bus_case, is_slack=False):

        self.id = int(bus_case[BUS_H['BUS_I']])
        self.type = int(bus_case[BUS_H['BUS_TYPE']])
        self.is_slack = is_slack
        self.v_max = bus_case[BUS_H['VMAX']]
        self.v_min = bus_case[BUS_H['VMIN']]

        if self.is_slack:
            self.v_slack = self.v_max

        self.v = None  # p.u (complex)
        self.p = 0  # MW
        self.q = 0  # MVAr

        self.p_min = None
        self.p_max = None
        self.q_min = None
        self.q_max = None