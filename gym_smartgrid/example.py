from gym_smartgrid.envs import SmartGridEnv6


if __name__ == '__main__':
    env = SmartGridEnv6()
    env.reset()
    for _ in range(50000):
        env.render(speed=2)
        env.step(env.action_space.sample()) # take a random action
    env.close()
