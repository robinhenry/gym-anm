import warnings
import numpy as np

from gym_anm.constants import BUS_H, DEV_H, BRANCH_H


def check_network_file(case):
    """
    Check that the input network file follows the required structure.

    Parameters
    ----------
    case : dict of numpy.ndarray

    Raises
    ------
    ValueError
        If the network file does not follow the required structure.
    NotImplementedError
        If the network file contains devices of unsupported type.
    """

    _check_buses(case['bus'])
    _check_branches(case['branch'])
    _check_devices(case['device'])


def _check_buses(buses):
    """ Check the structure of the input BUS array. """

    # Check that the array if of the right shape.
    if buses.shape[1] != 5:
        raise ValueError('network["BUS"] should have 5 columns.')

    # Check that there is exactly 1 slack bus.
    if np.sum(buses[:, BUS_H['BUS_TYPE']] == 3) != 1:
        raise ValueError('There should be exactly one bus with BUS_TYPE == 3 '
                         'in the network file, i.e. 1 slack bus.')

    # Check that all other buses are of type PQ.
    if np.sum(buses[:, BUS_H['BUS_TYPE']] == 1) != (buses.shape[0] - 1):
        raise ValueError('All buses, except the slack bus, should have '
                         'BUS_TYPE == 1 in the network file; only PV buses are '
                         'supported at this point.')

    # Check that the slack bus has a fixed voltage point.
    if buses[buses[:, BUS_H['BUS_TYPE']] == 3, BUS_H['VMAX']] != buses[
        buses[:, BUS_H['BUS_TYPE']] == 3, BUS_H['VMIN']]:
        raise ValueError('The slack bus should have VMAX == VMIN in the '
                         'network file.')


def _check_branches(branches):
    """ Check the structure of the input BRANCH array. """

    # Check that the array if of the right shape.
    if branches.shape[1] != 9:
        raise ValueError('network["BRANCH"] should have 9 columns.')

    # Warn the user if RATE == 0.
    for idx, branch in enumerate(branches):
        if branch[BRANCH_H['RATE']] < 0:
            raise ValueError(f'The rate of branch {idx} is < 0 in the network '
                             f'file.')
        if branch[BRANCH_H['RATE']] == 0:
            warnings.warn(f'The rate of branch {idx} is 0 in the network file.')


def _check_devices(devices):
    """ Check the structure of the input DEVICE array. """

    # Check that the array if of the right shape.
    if devices.shape[1] != 16:
        raise ValueError('network["DEVICE"] should have 16 columns.')

    # Check that there is exactly 1 slack device.
    if np.sum(devices[:, DEV_H['DEV_TYPE']] == 0) != 1:
        raise ValueError('There should only be 1 device with DEV_TYPE == 0 in '
                         'the network file, i.e. only 1 slack device.')

    # Check that no Power Plant device is specified - unsupported yet.
    if np.any(devices[:, DEV_H['DEV_TYPE']] == 1):
        raise NotImplementedError('The device type DEV_TYPE == 1 (power '
                                  'plant) is not supported yet.')

    for idx, dev in enumerate(devices):

        # Warn the user if a Q/P ratio is provided for slack or DES devices.
        if dev[DEV_H['DEV_TYPE']] == 0 or dev[DEV_H['DEV_TYPE']] == 4:
            if dev[DEV_H['Q/P']] != 0:
                warnings.warn(f'Device {idx}: the Q/P ratio value will not be '
                              f'used for device with DEV_TYPE == '
                              f'{dev[DEV_H["DEV_TYPE"]]}.')

        # Warn the user if QMAX, QMIN values are specified for load devices.
        if dev[DEV_H['DEV_TYPE']] == -1:
            if dev[DEV_H['QMAX']] != 0. or dev[DEV_H['QMIN']] != 0:
                warnings.warn(f'Device {idx}: a value for VMAX or VMIN has '
                              f'been specified in the network file but will '
                              f'not be used.')
