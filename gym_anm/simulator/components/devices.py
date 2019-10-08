import numpy as np

from gym_anm.constants import DEV_H


class Device(object):
    """
    A electric device connected to an electric power grid.

    Attributes
    ----------
    dev_id : int
        The unique device ID.
    type_id : int
        The unique ID within a subset of devices (e.g. loads, generators, etc.).
    bus_id : int
        The ID of the bus the device is connected to.
    type : int
        The device type : -1 (load), 0 (slack), 1 (power plant), 2 (wind),
        3 (solar), 4 (storage).
    qp_ratio : float
        The constant ratio of reactive over real power injection.
    p_min, p_max, q_min, q_max : float
        The minimum and maximum real (MW) and reactive (MVAr) power injections.
    pc1, pc2, qc1_min, qc1_max, qc2_min, qc2_max : float
        The bounds describing the (P, Q) region of operation of the device.
    soc_min, soc_max : float
        The minimum and maximum state of charge (MWh) if the device is a storage
        unit; None otherwise.
    eff : float
        The round-trip efficiency in [0, 1] if the device is a storage unit;
        None otherwise.
    lag_slope, lag_off : float
        The slope and offset of the lagging line constraining the (P, Q) region of
        operation of the device.
    lead_slope, lead_off : float
        The sole and offset of the leading line constraining the (P, Q) region of
        operation of the device.
    is_slack : bool
        True if the device is connected to the slack bus; False otherwise.
    p, q : float
        The current real (MW) and reactive (MVAr) power injection from the device.
    soc : float
        The state of charge if the device is a storage unit; None otherwise.


    Methods
    -------
    compute_pq(p_from, q_from)
        Compute the closest (P, Q) feasible power injection point.
    """

    def __init__(self, dev_id, dev_spec_id, dev_case):
        """
        Parameters
        ----------
        dev_id : int
            The device unique ID.
        dev_spec_id : int
            The unique ID specific to the subset of devices of the same type.
        dev_case : numpy.ndarray
            The corresponding device row in the network file describing the
            network.
        """

        self.dev_id = dev_id
        self.type_id = dev_spec_id
        self.bus_id = int(dev_case[DEV_H['BUS_I']])
        self.type = int(dev_case[DEV_H['DEV_TYPE']])
        self.qp_ratio = dev_case[DEV_H['Q/P']]
        self.q_max = dev_case[DEV_H['QMAX']]
        self.q_min = dev_case[DEV_H['QMIN']]
        self.p_max = dev_case[DEV_H['PMAX']]
        self.p_min = dev_case[DEV_H['PMIN']]
        self.pc1 = dev_case[DEV_H["PC1"]]
        self.pc2 = dev_case[DEV_H["PC2"]]
        self.qc1_min = dev_case[DEV_H["QC1MIN"]]
        self.qc1_max = dev_case[DEV_H["QC1MAX"]]
        self.qc2_min = dev_case[DEV_H["QC2MIN"]]
        self.qc2_max = dev_case[DEV_H["QC2MAX"]]

        if self.type == 4:
            self.soc_min = 0.
            self.soc_max = dev_case[DEV_H['SOC_MAX']]
            self.eff = dev_case[DEV_H['EFF']]
        else:
            self.soc_min, self.soc_max = None, None
            self.eff = None

        if not self.type:
            self.is_slack = True
        else:
            self.is_slack = False

        self._compute_lag_lead_limits(dev_case)

        self.p = None
        self.q = None
        self.soc = 0.

    def _compute_lag_lead_limits(self, dev_case):
        dQ_min = dev_case[DEV_H["QC2MIN"]] - dev_case[DEV_H["QC1MIN"]]
        dQ_max = dev_case[DEV_H["QC2MAX"]] - dev_case[DEV_H["QC1MAX"]]
        dP = dev_case[DEV_H["PC2"]] - dev_case[DEV_H["PC1"]]

        if dP and dev_case[DEV_H["PC1"]]:
            self.lag_slope = dQ_min / dP
            self.lag_off = dev_case[DEV_H["QC1MIN"]] \
                           - dQ_min / dP * dev_case[DEV_H["PC1"]]

            self.lead_slope = dQ_max / dP
            self.lead_off = dev_case[DEV_H["QC1MAX"]] \
                            - dQ_max / dP * dev_case[DEV_H["PC1"]]

        else:
            self.lag_slope, self.lag_off = None, None
            self.lead_slope, self.lead_off = None, None

    def compute_pq(self, p, q):
        """
        Compute the closest (P, Q) feasible power injection point.

        Parameters
        ----------
        p, q : float
            The desired (P, Q) power injection point (MW, MVAr).
        """
        raise NotImplementedError


class Load(Device):
    """
    A passive load connected to an electric power grid.
    """

    def __init__(self, dev_id, load_id, dev_case):
        """
        Parameters
        ----------
        dev_id : int
            The unique device ID.
        load_id : int
            The unique load ID.
        dev_case : numpy.ndarray
            The corresponding device row in the network file describing the
            network.
        """

        super().__init__(dev_id, load_id, dev_case)

        self.q_min = self.p_min * self.qp_ratio
        self.q_max = self.q_max * self.qp_ratio

    def compute_pq(self, p, q=None):
        # docstring inherited

        self.p  = p
        self.q = p * self.qp_ratio


class Generator(Device):
    """
    A generator connected to an electrical power grid.
    """

    def __init__(self, dev_id, gen_id, dev_case):
        """
        Parameters
        ----------
        dev_id : int
            The unique device ID.
        gen_id : int
            The unique generator ID.
        dev_case : numpy.ndarray
            The corresponding device row in the network file describing the
            network.
        """

        super().__init__(dev_id, gen_id, dev_case)

    def compute_pq(self, p_init, q_init=None):
        # docstring inherited

        p = p_init
        p = np.minimum(p, self.p_max)
        p = np.maximum(p, self.p_min)

        q = p * self.qp_ratio
        q = np.minimum(q, self.q_max)
        q = np.maximum(q, self.q_min)

        if (self.lead_slope is not None) and (self.lead_off is not None):
            if q > self.lead_slope * p + self.lead_off:
                p = self.lead_off / (self.qp_ratio - self.lead_slope)
                q = p * self.qp_ratio

        if (self.lag_slope is not None) and (self.lag_off is not None):
            if q < self.lag_slope * p + self.lag_off:
                p = self.lag_off / (self.qp_ratio - self.lag_slope)
                q = p * self.qp_ratio

        self.p = p
        self.q = q


class PowerPlant(Generator):
    """
    A non-renewable energy source connected to an electrical power grid.
    """

    def __init__(self, dev_id, gen_id, dev_case):
        # docstring inherited

        super().__init__(dev_id, gen_id, dev_case)


class VRE(Generator):
    """
    A renewable energy source connected to an electrical power grid.
    """

    def __init__(self, dev_id, gen_id, dev_case):
        # docstring inherited

        super().__init__(dev_id, gen_id, dev_case)


class Storage(Device):
    """
    A distributed storage unit connected to an electrical power grid.
    """

    def __init__(self, dev_id, su_id, dev_case):
        """
        Parameters
        ----------
        dev_id : int
            The unique device ID.
        su_id : int
            The unique storage unit iD.
        dev_case : numpy.ndarray
            The corresponding device row in the network file describing the
            network.
        """

        super().__init__(dev_id, su_id, dev_case)

    def manage(self, alpha_setpoint, delta_t, q_setpoint):
        """
        Compute the (P, Q) injection point and update the SoC.

        Parameters
        ----------
        alpha_setpoint : float
            The desired charging rate (MW).
        delta_t : float
            The fraction of an hour representing a single timestep, e.g. 0.25 for
            a timestep of 15 minutes.
        q_setpoint : float
            The desired reactive power injection into the network (MVAr).
        """

        # Truncate desired charging rate if out of operating bounds.
        max_alpha = np.minimum(alpha_setpoint, self.p_max)
        max_alpha = np.maximum(max_alpha, self.p_min)

        # Get energy being transferred.
        if alpha_setpoint > 0.:
            delta_soc = delta_t * (1 + self.eff) * max_alpha / 2.
        else:
            delta_soc = delta_t * max_alpha

        # Check that SoC constraints are satisfied and modify the charging
        # rate if it is.
        # Case 1: upper limit.
        if self.soc + delta_soc > self.soc_max:
            delta_soc = self.soc_max - self.soc
            max_alpha = 2 * delta_soc / (delta_t * (1 + self.eff))

        # Case 2: lower limit.
        elif self.soc + delta_soc < self.soc_min:
            delta_soc = self.soc_min - self.soc
            max_alpha = delta_soc / delta_t

        # Get the real power injection in the network.
        if alpha_setpoint > 0.:
            p = - max_alpha
        else:
            p = - 2 * self.eff * max_alpha / (1 + self.eff)

        # Compute the reactive power injection into the network.
        self.compute_pq(p, q_setpoint)

        # Update the state of charge.
        self.soc += delta_soc


    def compute_pq(self, p_init, q_init):
        # docstring inherited

        p_max = np.maximum(self.p_min, p_init)
        p_max = np.minimum(self.p_max, p_max)

        p = np.abs(p_max)
        q = q_init

        q = np.minimum(q, self.q_max)
        q = np.maximum(q, self.q_min)

        if q > self.lead_slope * p + self.lead_off:
            q = self.lead_slope * p + self.lead_off

        if q < self.lag_slope * p + self.lag_off:
            q = self.lag_slope * p + self.lag_off

        self.p = p_max
        self.q = q
