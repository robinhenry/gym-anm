"""Utility functions."""

from gym_anm.errors import ArgsError, ObsSpaceError, ObsNotSupportedError, UnitsNotSupportedError
from gym_anm.simulator.components.constants import STATE_VARIABLES


def check_env_args(K, delta_t, lamb, gamma, observation, aux_bounds,
                   state_bounds):
    """
    Raises an error if the arguments of the new environment do not match the
    requirements.

    Parameters
    ----------
    K : int
        The number of auxiliary variables.
    delta_t : float
        The interval of time between two consecutive time steps (fraction of
        hour).
    lamb : int or float
        The factor multiplying the penalty associated with violating
        operational constraints (used in the reward signal).
    gamma : float
        The discount factor in [0, 1].
    observation : callable or str or list
        The observation space. It can be specified as "state" to construct a
        fully observable environment (:math:`o_t = s_t`); as a callable function such
        that :math:`o_t = observation(s_t)`; or as a list of tuples (x, y, z) that
        refers to the electrical quantity x (str) at the nodes/branches/devices
        y (list) in unit z (str, optional).
    aux_bounds : np.ndarray
            The bounds on the auxiliary internal variables as a 2D array where
            the :math:`k^{th}-1` auxiliary variable is bounded by
            [aux_bounds[k, 0], aux_bounds[k, 1]]. This can be useful if auxiliary
            variables are to be included in the observation vectors and a bounded
            observation space is desired.
    state_bounds : dict of {str : dict}
        The bounds on the state variables of the distribution network.
    """

    if K < 0:
        raise ArgsError('The argument K is %d but should be >= 0.' % (K))
    if delta_t <= 0:
        raise ArgsError('The argument delta_t is %.2f but should be > 0.' % (delta_t))
    if lamb < 0:
        raise ArgsError('The argument lamb is %d but should be >= 0.' % (lamb))
    if gamma < 0 or gamma > 1:
        raise ArgsError('The argument gamma is %.4f but should be in [0, 1].' % (gamma))

    # Check that the observation space is correctly specified.
    if isinstance(observation, str) and observation == 'state':
        pass
    elif isinstance(observation, list):
        _check_observation_vars(observation, state_bounds, K)
    elif callable(observation):
        pass
    else:
        raise ArgsError('The argument observation is of type {} but should be '
                        'either a list, a callable, or the string "state".'
                        .format(type(observation)))

    # Check that aux_bounds is correctly specified.
    if aux_bounds is not None:
        if len(aux_bounds) != K:
            raise ArgsError('The argument aux_bounds has length {} but the '
                            'environment has K={} auxiliary variables.'
                            .format(len(aux_bounds), K))


def _check_observation_vars(observation, state_bounds, K):
    """
    Checks the specs of the observation space when specified as a list.

    Parameters
    ----------
    observation : list of tuples
        The observation space as a list of tuples (x, y, z), each of which
        refers to the electrical quantity x (str) at the nodes/branches/devices
        y (list) in unit z (str, optional).
    state_bounds : dict of {str : dict}
        The bounds on the state variables of the distribution network.
    K : int
        The number of auxiliary variables.
    """

    for obs in observation:
        if len(obs) not in [2, 3]:
            raise ObsSpaceError('The observation tuple {} should be a list with '
                                '2 or 3 elements.'.format(obs))

        # Check that the observation variable is supported.
        key = obs[0]
        if key not in STATE_VARIABLES.keys():
            raise ObsNotSupportedError(key, STATE_VARIABLES)

        # Check that the nodes/devices/branches specified exist.
        nodes = obs[1]
        if isinstance(nodes, str) and nodes == 'all':
            pass
        elif key == 'aux':
            for n in nodes:
                if n >= K:
                    raise ObsSpaceError('Aux variable index {} is out of bound '
                                        'for {} aux variables.'.format(n, K))
        elif isinstance(nodes, list):
            for n in nodes:
                if n not in state_bounds[key].keys():
                    raise ObsSpaceError('Observation {} is not supported for '
                                        'device/branch/bus with ID {}.'
                                        .format(key, n))
        else:
            raise ObsSpaceError()

        # Check that the unit specified is supported.
        if len(obs) == 3:
            units = obs[2]
            if units not in STATE_VARIABLES[key]:
                raise UnitsNotSupportedError(units, STATE_VARIABLES[key], key)