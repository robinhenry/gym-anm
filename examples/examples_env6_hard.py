import numpy as np

from gym_anm.envs.anm6_env.anm6_hard import ANM6Hard


def null_agent():
    return np.array([30, 50, 0, 0])


if __name__ == '__main__':
    env = ANM6Hard()
    obs = env.reset()

    for i in range(1000):
        env.render(sleep_time=.5)

        a = null_agent()
        obs, r, done, info = env.step(a)
        print('Reward: ', r)

    env.close()