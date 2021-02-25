import numpy as np

from .components.constants import BUS_H, DEV_H, BRANCH_H
from .components.errors import BusSpecError, DeviceSpecError, BaseMVAError, BranchSpecError


def check_network_specs(network):
    """
    Check that the input network is properly specified.

    Parameters
    ----------
    network : dict of numpy.ndarray
        The dictionary containing the specs of the distribution network.
    """

    # Check base MVA is > 0.
    if network['baseMVA'] <= 0.:
        raise BaseMVAError()

    # Check that a unique slack bus is specified.
    bus_types = network['bus'][:, BUS_H['BUS_TYPE']]
    if np.sum(bus_types == 0) != 1:
        raise BusSpecError('The network bus array should contain exactly 1 slack bus (with BUS_TYPE = 0).')

    # Check that there is a unique slack device.
    dev_types = network['device'][:, DEV_H['DEV_TYPE']]
    if np.sum(dev_types == 0) != 1:
        raise DeviceSpecError('The network device array should contain exactly 1 slack device (with DEV_TYPE = 0).')

    # Check slack bus and slack device match.
    slack_bus = network['bus'][np.where(np.array(bus_types) == 0)[0], BUS_H['BUS_ID']]
    bus_of_dev = network['device'][np.where(np.array(dev_types) == 0)[0], DEV_H['BUS_ID']]
    if slack_bus != bus_of_dev:
        raise DeviceSpecError('The slack bus ID of the slack device is %d but the actual slack bus has ID %d.'
                              % (bus_of_dev, slack_bus))

    # Check buses have unique IDs.
    bus_ids = network['bus'][:, BUS_H['BUS_ID']]
    set_ids = set(bus_ids)
    if len(bus_ids) != len(set_ids):
        raise BusSpecError('The buses should all have unique IDs.')

    # Check devices have unique IDs.
    dev_ids = network['device'][:, DEV_H['DEV_ID']]
    set_ids = set(dev_ids)
    if len(dev_ids) != len(set_ids):
        raise DeviceSpecError('The devices should all have unique IDs.')

    # Check that each branch is unique (parallel branches not supported).
    set_branches = set()
    for br_spec in network['branch']:
        f_bus = br_spec[BRANCH_H['F_BUS']]
        t_bus = br_spec[BRANCH_H['T_BUS']]
        set_branches.add((f_bus, t_bus))
        set_branches.add((t_bus, f_bus))
    if len(set_branches) != 2 * network['branch'].shape[0]:
        raise BranchSpecError('The network cannot contain parallel branches.')

    # Check that each branch links existing buses.
    for br in set_branches:
        if br[0] not in bus_ids or br[1] not in bus_ids:
            raise BranchSpecError('Existing buses are {} but you have a branch {}'
                                  .format(bus_ids, br))

