
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
