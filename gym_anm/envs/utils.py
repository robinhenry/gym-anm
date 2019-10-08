from collections import Iterable

from gym_anm.constants import OBSERVATION_VALUES


def sample_action(np_random, action_space):
    """
    Sample a random action from the action space.

    Parameters
    ----------
    np_random : numpy.random.RandomState
        The random seed to use. This should be the one used by the environment.
    action_space : gym.spaces.Tuple
        The action space of the environment.

    Returns
    -------
    gym.spaces.Tuple
        An action randomly selected from the action space.
    """

    actions = []
    for space in action_space:
        a = np_random.uniform(space.low, space.high, space.shape)
        actions.append(a)

    return tuple(actions)


def check_obs_values(obs_values):
    """
    Check that the observation space specified is valid.

    This function checks that the observation spaced specified in the
    implementation of a new gym-anm environment is valid, i.e.:
        1. Is given as a list of strings,
        2. All type of observations are supported by the simulator.

    Parameters
    ----------
    obs_values : list of str
        The types of observations that the environment should emit.

    Raises
    ------
    TypeError
        If `obs_values` is not a list.
    ValueError
        If any observation in `obs_values` is not supported by the simulator.

    """

    # Check that obs_values is a list.
    if not isinstance(obs_values, list):
        raise TypeError('The observation space "obs_values" should be '
                         'specified as a list.')

    # Check that each type of observation is supported.
    for v in obs_values:
        if v not in OBSERVATION_VALUES:
            raise ValueError('The observation type ' + v + ' is not supported.')


def check_init_soc(soc_init, soc_max):
    """
    Check that the implementation of init_soc() is valid.

    This function checks the output of the function `init_soc(soc_max)`,
    which is used to initialize the SoC of each DES unit. It checks:
        1. If `soc_init` is a list of the same size as `soc_max`, or is None,
        2. If the initial SoC of each DES unit lies in the range [0, c_max].

    Parameters
    ----------
    soc_init : list of float
        The initial SoC returned by the function `init_soc(soc_max)`.
    soc_max : list of float
        The maximum SoC of each DES unit.

    Raises
    ------
    TypeError
        If `soc_init` is neither a list nor None.
    ValueError
        If `soc_init` is a list but of the wrong size, or some initial SoC's
        are out of bounds.
    """

    if isinstance(soc_init, list):

        # Check if soc_init is of the right size.
        if len(soc_init) != len(soc_max):
            raise ValueError(f'The return value of init_soc(soc_max) should '
                             f'contain {len(soc_max)} values.')

        # Check that each initial SoC lies in [0, c_max].
        for i in range(len(soc_max)):
            if soc_init[i] < 0 or soc_init[i] > soc_max[i]:
                raise ValueError(f'soc_init[{i}] is {soc_init[i]} but should be in '
                                 f'the range [0, {soc_max[i]}.')

    elif soc_init is None:
        pass

    else:
        raise TypeError('The return value of init_soc(soc_max) should be a '
                        'list or None.')


def check_load_dg_iterators(iterators, dev_specs):
    """
    Check that the implementation of init_load_dg() is valid.

    This function checks if the return value of init_load_dg() meets some
    basic criteria, specifically:
        1. It is a list of the corresponding size,
        2. Each object in the list is iterable.

    Parameters
    ----------
    iterators : list of object
        The iterator objects returned by init_load_dg().
    dev_specs : list of (int, float)
        A list with a pair of (dev_type, p_max) for each device that requires
        the implementation of a stochastic process.

    Raises
    ------
    ValueError
        If the number of iterators is wrong.
    TypeError
        If any of the iterators is not iterable.
    """

    n_stoc_processes = len(dev_specs)

    if len(iterators) != n_stoc_processes:
        raise ValueError(f'The return value of init_dg_load(...) should be a '
                         f'list of size {n_stoc_processes}.')

    for idx, it in enumerate(iterators):
        if not isinstance(it, Iterable):
            raise TypeError(f'Object {idx} is not iterable.')
