from gym_smartgrid.envs import SmartGridEnv6


if __name__ == '__main__':
    env = SmartGridEnv6()
    env.reset()
    for _ in range(50):
        env.render('save')
        env.step(env.action_space.sample()) # take a random action
    history = env.close(path='test_history.csv')

    env.replay('test_history.csv')
