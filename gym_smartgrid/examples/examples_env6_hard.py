import numpy as np

from gym_smartgrid.envs import SmartGrid6Hard


def null_agent():
    return np.array([30, 50]), np.array([0]), np.array([0])


if __name__ == '__main__':
    env = SmartGrid6Hard()
    obs = env.reset()

    for i in range(1000):
        env.render(sleep_time=.5)

        a = null_agent()
        obs, r, done, info = env.step(a)
        print('Reward: ', r)

    env.close()