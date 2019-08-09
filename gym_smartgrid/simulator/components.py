import numpy as np

from gym_smartgrid.simulator.case_headers import BUS_H, BRANCH_H, DEV_H


class Bus(object):
    def __init__(self, bus_case, is_slack=False):
        self.id = bus_case[BUS_H['BUS_I']]
        self.type = bus_case[BUS_H['BUS_TYPE']]
        self.is_slack = is_slack

        if self.is_slack:
            self.v_slack = self.v_max
        else:
            self.v_max = bus_case[BUS_H['VMAX']]
            self.v_min = bus_case[BUS_H['VMIN']]

        self.v = None  # p.u (complex)
        self.p = 0  # MW
        self.q = 0  # MVAr

        self.p_min = None
        self.p_max = None
        self.q_min = None
        self.q_max = None



class TransmissionLine(object):
    def __init__(self, br_case):
        self.f_bus = br_case[BRANCH_H['F_BUS']]
        self.t_bus = br_case[BRANCH_H['T_BUS']]
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


class Device(object):
    def __init__(self, dev_id, dev_spec_id, dev_case):
        self.dev_id = dev_id
        self.type_id = dev_spec_id
        self.bus_id = dev_case[DEV_H['BUS_I']]
        self.type = dev_case[DEV_H['DEV_TYPE']]
        self.qp_ratio = dev_case[DEV_H['Q/P']]
        self.q_max = dev_case[DEV_H['QMAX']]
        self.q_min = dev_case[DEV_H['QMIN']]
        self.p_max = dev_case[DEV_H['PMAX']]
        self.p_min = dev_case[DEV_H['PMIN']]
        self.soc_max = dev_case[DEV_H['SOC_MAX']]
        self.eff = dev_case[DEV_H['EFF']]

        self._compute_lag_lead_limits()

        self.p = None  # MW
        self.q = None  # MVAr

    def _compute_lag_lead_limits(self, dev_case):
        dQ_min = dev_case["QC2MIN"] - dev_case["QC1MIN"]
        dQ_max = dev_case["QC2MAX"] - dev_case["QC1MAX"]
        dP = dev_case["PC2"] - dev_case["PC1"]

        if dP and dev_case["PC1"]:
            self.lag_slope = dQ_min / dP
            self.lag_off = dev_case["QC1MIN"] - dQ_min / dP * dev_case["PC1"]

            self.lead_slope = dQ_max / dP
            self.lead_off = dev_case["QC1MAX"] - dQ_max / dP * dev_case["PC1"]

        else:
            self.lag_slope, self.lag_off = None, None
            self.lead_slope, self.lead_off = None, None

    def compute_pq(self, p, q):
        raise NotImplementedError


class Load(Device):
    def __init__(self, dev_id, load_id, dev_case):
        super().__init__(dev_id, load_id, dev_case)

    def compute_pq(self, p, q=None):
        self.p  = p
        self.q = p * self.qp_ratio



class Generator(Device):
    def __init__(self, dev_id, gen_id, dev_case, is_slack):
        super().__init__(dev_id, gen_id, dev_case)
        self.is_slack = is_slack

    def compute_pq(self, p_init, q_init=None):
        p = p_init
        p = np.minimum(p, self.p_max)
        p = np.maximum(p, self.p_min)

        q = p * self.qp_ratio
        q = np.minimum()(q, self.q_max)
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
    def __init__(self, dev_id, gen_id, dev_case, is_slack=False):
        super().__init__(dev_id, gen_id, dev_case, is_slack)


class VRE(Generator):
    def __init__(self, dev_id, gen_id, dev_case, is_slack=False):
        super().__init__(dev_id, gen_id, dev_case, is_slack)


class Storage(Device):
    def __init__(self, dev_id, su_id, dev_case):
        super().__init__(dev_id, su_id, dev_case)
        self.soc = None
        self.soc_min = 0.

    def manage(self, desired_alpha, delta_t, q_setpoint):

        # Truncate desired charging rate if above upper limit.
        max_alpha = np.minimum(desired_alpha, self.p_max)
        max_alpha = np.maximum(max_alpha, self.p_min)

        # Get energy being transferred.
        if desired_alpha > 0.:
            delta_soc = delta_t * (1 + self.eff) * max_alpha / 2.
        else:
            delta_soc = delta_t * max_alpha

        # Check that SoC constraints are satisfied and modify the charging
        # rate if it is.
        # Case 1: upper limit.
        if self.soc + delta_soc > self.soc_max:
            delta_soc = self.soc_max - self.soc
            max_alpha = 2 * delta_soc / (delta_t * (1 + eff))

        # Case 2: lower limit.
        elif self.soc + delta_soc < self.soc_min:
            delta_soc = self.soc - self.soc_min
            max_alpha = delta_soc / delta_t

        # Get the real power injection in the network.
        if desired_alpha > 0.:
            p = - max_alpha
        else:
            p = - 2 * self.eff * max_alpha / (1 + self.eff)

        self.compute_pq(p, q_setpoint)
        self.soc += delta_soc


    def compute_pq(self, p_init, q_init):
        p = np.abs(p_init)
        q = q_init

        q = np.minimum(q, self.q_max)
        q = np.maximum(q, self.q_min)

        if q > self.lead_slope * p + self.lead_off:
            q = self.lead_slope * p + self.lead_off

        if q < self.lag_slope * p + self.lag_off:
            q = self.lag_slope * p + self.lag_off

        self.p = p_init
        self.q = q_init
