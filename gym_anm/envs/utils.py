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

    if not isinstance(obs_values, list):
        raise TypeError('The observation space "obs_values" should be '
                         'specified as a list.')

    for v in obs_values:
        if v not in OBSERVATION_VALUES:
            raise ValueError('The observation type ' + v + ' is not supported.')


def check_init_soc(soc_init, soc_max):

    if isinstance(soc_init, list):

        if len(soc_init) != len(soc_max):
            raise ValueError(f'The return value of init_soc(soc_max) should '
                             f'contain {len(soc_max)} values.')

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
    n_stoc_processes = len(dev_specs)

    if len(iterators) != n_stoc_processes:
        raise ValueError(f'The return value of init_dg_load(...) should be a '
                         f'list of size {n_stoc_processes}.')
