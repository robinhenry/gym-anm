import numpy as np
import cvxpy as cp
from logging import getLogger

from .constants import DEV_H
from .errors import DeviceSpecError, LoadSpecError, GenSpecError, StorageSpecError

logger = getLogger(__file__)


class Device(object):
    """
    A electric device connected to an electric power grid.

    Attributes
    ----------
    dev_id : int
        The unique device ID :math:`d`.
    bus_id : int
        The ID of the bus the device is connected to :math:`i`.
    type : int
        The device type : -1 (load), 0 (slack), 1 (classical generator),
        2 (renewable energy), 3 (storage).
    qp_ratio : float
        The constant ratio of reactive over real power injection :math:`(Q/P)_d`.
    p_min, p_max, q_min, q_max : float
        The minimum and maximum real (p.u.) and reactive (p.u.) power injections
        (:math:`\\underline P_d, \\overline P_d, \\underline Q_d, \\overline Q_d`).
    p_plus, p_minus, q_plus, q_minus : float
        The bounds describing the (P, Q) region of physical operation of the device (p.u.)
        (:math:`P^+_d, P^-_d, Q^+_d, Q^-_d`).
    soc_min, soc_max : float
        The minimum and maximum state of charge if the device is a storage
        unit :math:`[\\underline {SoC}_d, \\overline {SoC}_d]` (p.u. per hour); None otherwise.
    eff : float
        The round-trip efficiency :math:`\\eta \\in [0, 1]` if the device is a storage unit;
        None otherwise.
    tau_1, tau_2, tau_3, tau_4 : float
        The slope of the constraints on Q when P is near its maximum (:math:`\\tau^{(1)}_d,
        \\tau^{(2)}_d, \\tau^{(3)}_d, \\tau^{(4)}_d`) (1-2 are used for generators, 1-4
        for storage units, and the remaining ones are None).
    rho_1, rho_2, rho_3, rho_4 : float
        The offset of the linear constraints on Q when P is near its maximum
        (:math:`\\rho^{(1)}_d, \\rho^{(2)}_d, \\rho^{(3)}_d, \\rho^{(4)}_d`)(see `tau_1`).
    is_slack : bool
        True if the device is connected to the slack bus; False otherwise.
    p, q : float
        The real :math:`P_d^{(dev)}` and reactive :math:`Q_d^{(dev)}` power injections from the device (p.u.).
    p_pot : float
        The maximum active generation of the device at the current time :math:`P_g^{(max)}` (p.u.),
        if a generator; None otherwise.
    soc : float
        The state of charge :math:`SoC_d` if the device is a storage unit (p.u. per hour);
        None otherwise.
    """

    def __init__(self, dev_spec, bus_ids, baseMVA):
        """
        Parameters
        ----------
        dev_spec : numpy.ndarray
            The corresponding device row in the network file describing the
            network.
        bus_ids : list of int
            The list of unique bus IDs.
        baseMVA : int
            The base power of the network.
        """

        # Components used by all types of devices.
        self.dev_id = dev_spec[DEV_H['DEV_ID']]
        if self.dev_id is None:
            raise DeviceSpecError('The device ID cannot be None.')
        else:
            self.dev_id = int(self.dev_id)

        self.bus_id = dev_spec[DEV_H['BUS_ID']]
        if self.bus_id is None or self.bus_id not in bus_ids:
            raise DeviceSpecError('Device {} has unique bus ID = {} but should'
                                  ' be in {}.'.format(self.dev_id, self.bus_id,
                                                      bus_ids))
        else:
            self.bus_id = int(self.bus_id)

        self.type = dev_spec[DEV_H['DEV_TYPE']]
        allowed = [-1, 0, 1, 2, 3]
        if self.type is None or self.type not in allowed:
            raise DeviceSpecError('The DEV_TYPE value for device %d should be'
                                  ' in %s.' % allowed)

        if self.type == 0:
            self.is_slack = True
        else:
            self.is_slack = False

        # Components device-specific.
        self.qp_ratio = None
        self.p_min, self.p_max = None, None
        self.q_min, self.q_max = None, None
        self.p_plus, self.p_minus = None, None
        self.q_plus, self.q_minus = None, None
        self.soc_min, self.soc_max = None, None
        self.eff = None
        self.tau_1, self.tau_2, self.tau_3, self.tau_4 = None, None, None, None
        self.rho_1, self.rho_2, self.rho_3, self.rho_4 = None, None, None, None

        self._check_type_specific_specs(dev_spec, baseMVA)

        self.p = None
        self.q = None
        self.soc = None      # only for storage units

    def _check_type_specific_specs(self, dev_spec, baseMVA):
        pass


class Load(Device):
    """
    A passive load connected to an electric power grid.
    """

    def __init__(self, dev_spec, bus_ids, baseMVA):
        # docstring inherited

        super().__init__(dev_spec, bus_ids, baseMVA)

    def _check_type_specific_specs(self, dev_spec, baseMVA):

        if self.type != -1:
            raise LoadSpecError('Trying to create a Load object for a device'
                                ' that has DEV_TYPE != -1.')

        self.qp_ratio = dev_spec[DEV_H['Q/P']]
        if self.qp_ratio is None:
            raise LoadSpecError('A fixed Q/P value must be provided for device'
                                ' (load) %d.' % self.dev_id)

        self.p_max = dev_spec[DEV_H['PMAX']]
        if self.p_max is None:
            self.p_max = 0.
        elif self.p_max > 0.:
            raise LoadSpecError('Trying to create a load with P_max > 0. Loads'
                                ' can only withdraw power from the grid.')

        self.p_min = dev_spec[DEV_H['PMIN']]
        if self.p_min is None:
            self.p_min = - np.inf
            logger.warning('The P_min value of device %d is set to - infinity.'
                           % self.dev_id)
        else:
            self.p_min /= baseMVA  # to p.u.

        if self.p_max < self.p_min:
            raise LoadSpecError('Device %d has P_max < P_min.' % self.dev_id)

        self.q_max = self.p_max * self.qp_ratio
        self.q_min = self.p_min * self.qp_ratio

    def map_pq(self, p):
        """
        Map p to the closest (P, Q) feasible power injection point.

        Parameters
        ----------
        p : float
            The desired :math:`P_l` power injection point (p.u.).
        """

        self.p = np.clip(p, self.p_min, self.p_max)
        self.q = self.p * self.qp_ratio


class Generator(Device):
    """
    A generator connected to an electrical power grid.
    """

    def __init__(self, dev_spec, bus_ids, baseMVA):
        # docstring inherited

        super().__init__(dev_spec, bus_ids, baseMVA)
        self.p_pot = self.p_max

    @property
    def p_pot(self):
        return self._p_pot

    @p_pot.setter
    def p_pot(self, value):
        self._p_pot = np.clip(value, self.p_min, self.p_max)

    def _check_type_specific_specs(self, dev_spec, baseMVA):

        self.p_max = dev_spec[DEV_H['PMAX']]
        if self.p_max is None:
            self.p_max = np.inf
            logger.warning('The P_max value of device %d is set to infinity.'
                           % self.dev_id)
        elif self.p_max < 0 and not self.is_slack:
            raise GenSpecError(
                'The PMAX value of device %d should be >= 0.' % self.dev_id)
        else:
            self.p_max /= baseMVA  # to p.u.

        self.p_min = dev_spec[DEV_H['PMIN']]
        if self.p_min is None:
            if self.is_slack:
                self.p_min = - np.inf
            else:
                self.p_min = 0.
        elif self.p_min < 0 and not self.is_slack:
            raise GenSpecError(
                'The PMIN value of device %d should be >= 0.' % self.dev_id)
        else:
            self.p_min /= baseMVA  # to p.u.

        if self.p_max < self.p_min:
            raise GenSpecError('Device %d has PMAX < PMIN.' % self.dev_id)

        self.q_max = dev_spec[DEV_H['QMAX']]
        if self.q_max is None:
            self.q_max = np.inf
            if not self.is_slack:
                logger.warning('The Q_max value of device %d is set to '
                               'infinity.' % self.dev_id)
        else:
            self.q_max /= baseMVA  # to p.u.

        self.q_min = dev_spec[DEV_H['QMIN']]
        if self.q_min is None:
            self.q_min = - np.inf
            if not self.is_slack:
                logger.warning('The Q_min value of device %d is set to - '
                               'infinity.' % self.dev_id)
        else:
            self.q_min /= baseMVA  # to p.u.

        if self.q_max < self.q_min:
            raise GenSpecError('Device %d has QMAX < QMIN.' % self.dev_id)

        self.p_plus = dev_spec[DEV_H['P+']]
        if self.p_plus is None:
            self.p_plus = self.p_max
        else:
            self.p_plus /= baseMVA

        if self.p_plus < self.p_min:
            raise GenSpecError('Device %d has P+ < PMIN' % self.dev_id)
        elif self.p_plus > self.p_max:
            raise GenSpecError('Device %d has P+ > PMAX.' % self.dev_id)

        self.p_minus = dev_spec[DEV_H['P-']]
        if self.p_minus is not None:
            logger.warning('The P- value of device %d is going to be ignored.'
                           % self.dev_id)
            self.p_minus = None

        self.q_plus = dev_spec[DEV_H['Q+']]
        if self.q_plus is None:
            self.q_plus = self.q_max
        else:
            self.q_plus /= baseMVA

        if self.q_plus > self.q_max:
            raise GenSpecError('Device %d has Q+ > QMAX.' % self.dev_id)

        self.q_minus = dev_spec[DEV_H['Q-']]
        if self.q_minus is None:
            self.q_minus = self.q_min
        else:
            self.q_minus /= baseMVA

        if self.q_minus < self.q_min:
            raise GenSpecError('Device %d has Q- < QMIN.' % self.dev_id)

        if self.q_plus < self.q_minus:
            raise GenSpecError('Device %d has Q+ < Q-.' % self.dev_id)

        if self.p_max == self.p_plus:
            self.tau_1 = 0.
            self.tau_2 = 0.
        else:
            self.tau_1 = (self.q_plus - self.q_max) / (self.p_max - self.p_plus)
            self.tau_2 = (self.q_minus - self.q_min) / (self.p_max - self.p_plus)

        self.rho_1 = self.q_max - self.tau_1 * self.p_plus
        self.rho_2 = self.q_min - self.tau_2 * self.p_plus

    def map_pq(self, p, q):
        """
        Map (p, q) to the closest (P, Q) feasible power injection point.

        Parameters
        ----------
        p, q : float
            The desired (P, Q) power injection point (p.u.).
        """

        # Desired set-point.
        point = np.array([p, q])

        # Inequality constraints for the optimization problem.
        G = np.array([[-1, 0],
                      [1, 0],
                      [1, 0],
                      [0, -1],
                      [0, 1],
                      [-self.tau_1, 1],
                      [self.tau_2, -1]])

        h = np.array([-self.p_min,
                      self.p_max,
                      self.p_pot,
                      - self.q_min,
                      self.q_max,
                      self.rho_1,
                      - self.rho_2])

        # Define and solve the CVXPY problem.
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - point)),
                          [G @ x <= h])
        prob.solve()

        self.p = x.value[0]
        self.q = x.value[1]


class ClassicalGen(Generator):
    """
    A non-renewable energy source connected to an electrical power grid.
    """

    def __init__(self, dev_spec, bus_ids, baseMVA):
        # docstring inherited

        super().__init__(dev_spec, bus_ids, baseMVA)


class RenewableGen(Generator):
    """
    A renewable energy source connected to an electrical power grid.
    """

    def __init__(self, dev_spec, bus_ids, baseMVA):
        # docstring inherited

        super().__init__(dev_spec, bus_ids, baseMVA)


class StorageUnit(Device):
    """
    A distributed storage unit connected to an electrical power grid.
    """

    def __init__(self, dev_spec, bus_ids, baseMVA):

        super().__init__(dev_spec, bus_ids, baseMVA)

    def _check_type_specific_specs(self, dev_spec, baseMVA):

        self.p_max = dev_spec[DEV_H['PMAX']]
        if self.p_max is None:
            logger.warning('The P_max value of device %d is set to infinity.'
                           % (self.dev_id))
            self.p_max = np.inf
        elif self.p_max < 0:
            raise StorageSpecError(
                'The PMAX value of device %d should be >= 0.' % self.dev_id)
        else:
            self.p_max /= baseMVA  # to p.u.

        self.p_min = dev_spec[DEV_H['PMIN']]
        if self.p_min is None:
            self.p_min = - np.inf
            logger.warning('The P_min value of device %d is set to - infinity.'
                           % (self.dev_id))
        elif self.p_min > 0:
            raise StorageSpecError(
                'The PMIN value of device %d should be <= 0.' % self.dev_id)
        else:
            self.p_min /= baseMVA  # to p.u.

        if self.p_max < self.p_min:
            raise StorageSpecError('Device %d has PMAX < PMIN.' % self.dev_id)

        self.q_max = dev_spec[DEV_H['QMAX']]
        if self.q_max is None:
            logger.warning('The Q_max value of device %d is set to infinity.'
                           % self.dev_id)
            self.q_max = np.inf
        else:
            self.q_max /= baseMVA  # to p.u.

        self.q_min = dev_spec[DEV_H['QMIN']]
        if self.q_min is None:
            self.q_min = - np.inf
            logger.warning('The Q_min value of device %d is set to - infinity.'
                           % self.dev_id)
        else:
            self.q_min /= baseMVA  # to p.u.

        if self.q_max < self.q_min:
            raise StorageSpecError('Device %d has QMAX < QMIN.' % self.dev_id)

        self.p_plus = dev_spec[DEV_H['P+']]
        if self.p_plus is None:
            self.p_plus = self.p_max
        else:
            self.p_plus /= baseMVA

        if self.p_plus < self.p_min:
            raise StorageSpecError('Device %d has P+ < PMIN' % self.dev_id)
        elif self.p_plus > self.p_max:
            raise StorageSpecError('Device %d has P+ > PMAX.' % self.dev_id)

        self.p_minus = dev_spec[DEV_H['P-']]
        if self.p_minus is None:
            self.p_minus = self.p_min
        else:
            self.p_minus /= baseMVA

        if self.p_minus < self.p_min:
            raise StorageSpecError('Device %d has P- < PMIN' % self.dev_id)
        elif self.p_minus > self.p_max:
            raise StorageSpecError('Device %d has P- > PMAX' % self.dev_id)

        if self.p_plus < self.p_minus:
            raise StorageSpecError('Device %d has P+ < P-.' % self.dev_id)

        self.q_plus = dev_spec[DEV_H['Q+']]
        if self.q_plus is None:
            self.q_plus = self.q_max
        else:
            self.q_plus /= baseMVA

        if self.q_plus > self.q_max:
            raise StorageSpecError('Device %d has Q+ > QMAX.' % self.dev_id)

        self.q_minus = dev_spec[DEV_H['Q-']]
        if self.q_minus is None:
            self.q_minus = self.q_min
        else:
            self.q_minus /= baseMVA

        if self.q_minus < self.q_min:
            raise StorageSpecError('Device %d has Q- < QMIN.' % self.dev_id)

        if self.q_plus < self.q_minus:
            raise StorageSpecError('Device %d has Q+ < Q-.' % self.dev_id)

        if self.p_max == self.p_plus:
            self.tau_1 = 0.
            self.tau_2 = 0.
        else:
            self.tau_1 = (self.q_plus - self.q_max) / (self.p_max - self.p_plus)
            self.tau_2 = (self.q_minus - self.q_min) / (self.p_max - self.p_plus)

        self.rho_1 = self.q_max - self.tau_1 * self.p_plus
        self.rho_2 = self.q_min - self.tau_2 * self.p_plus

        if self.p_min == self.p_minus:
            self.tau_3 = 0.
            self.tau_4 = 0.
        else:
            self.tau_3 = (self.q_min - self.q_minus) / (self.p_minus - self.p_min)
            self.tau_4 = (self.q_max - self.q_plus) / (self.p_minus - self.p_min)

        self.rho_3 = self.q_min - self.tau_3 * self.p_minus
        self.rho_4 = self.q_max - self.tau_4 * self.p_minus

        self.soc_min = dev_spec[DEV_H['SOC_MIN']]
        if self.soc_min is None:
            self.soc_min = 0.
        elif self.soc_min < 0.:
            raise StorageSpecError('Device %d is a storage unit and has SOC_MIN'
                                   ' < 0.' % self.dev_id)
        else:
            self.soc_min /= baseMVA  # to p.u.

        self.soc_max = dev_spec[DEV_H['SOC_MAX']]
        if self.soc_max is None:
            raise StorageSpecError('Device %d is a storage unit and has SOC_MAX'
                                   ' = None.' % self.dev_id)
        if self.soc_max < 0.:
            raise StorageSpecError('Device %d is a storage unit and has SOC_MAX'
                                   ' < 0.' % self.dev_id)
        else:
            self.soc_max /= baseMVA  # to p.u.

        if self.soc_max < self.soc_min:
            raise StorageSpecError('Device %d is a storage unit and has SOC_MAX'
                                   ' < SOC_MIN.' % self.dev_id)

        self.eff = dev_spec[DEV_H['EFF']]
        if self.eff is None:
            logger.warning('Setting (dis)charging efficiency of device %d '
                           '(storage unit) to 1.' % self.dev_id)
            self.eff = 1.
        elif self.eff < 0. or self.eff > 1.:
            raise StorageSpecError('Device %d is a storage unit and has EFF '
                                   'outside of [0, 1].' % self.dev_id)

        return

    def map_pq(self, p, q, delta_t):
        """
        Map (p, q) to the closest (P, Q) feasible power injection point.

        Parameters
        ----------
        p, q : float
            The desired (P, Q) power injection point (p.u.).
        """

        # Desired set-point.
        point = np.array([p, q])

        # Inequality constraints.
        G = np.array([[-1, 0],
                      [1, 0],
                      [0, -1],
                      [0, 1],
                      [-self.tau_1, 1],
                      [self.tau_2, -1],
                      [self.tau_3, -1],
                      [-self.tau_4, 1],
                      [-1, 0],
                      [1, 0]])

        h = np.array([-self.p_min,
                      self.p_max,
                      -self.q_min,
                      self.q_max,
                      self.rho_1,
                      -self.rho_2,
                      -self.rho_3,
                      self.rho_4,
                      -(self.soc - self.soc_max) / (delta_t * self.eff),
                      self.eff * (self.soc - self.soc_min) / delta_t])

        # Define and solve the CVXPY problem.
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - point)), [G @ x <= h])
        prob.solve()

        self.p = x.value[0]
        self.q = x.value[1]

    def update_soc(self, delta_t):
        """
        Update the state of charge based on the current power injection.

        This function updates the new state of charge :math:`SoC_{t+1}`, assuming that
        the device injects :code:`self.p` active power (in p.u.) into the network
        during :code:`delta_t` hours.

        Parameters
        ----------
        delta_t : float
            Time interval between subsequent timesteps :math:`\\Delta t` (in fraction of hour).
        """

        # Compute the new SoC.
        if self.p <= 0:
            self.soc -= delta_t * self.eff * self.p
        else:
            self.soc -= delta_t * self.p / self.eff

        # Clip the new state of charge to be in [soc_min, soc_max].
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)
