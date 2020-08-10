import time

from gym_anm.envs import ANM6Easy
from gym_anm.dc_opf import DCOPFAgent


def run_baselines():
    env = ANM6Easy()
    gamma = env.gamma
    T = int(1000)

    for horizon in [1, 4, 8, 12, 16]:
        env.seed(1000)
        env.reset()

        agent = DCOPFAgent(env.simulator, env.action_space,
                           planning_steps=horizon)
        ret = 0.

        for i in range(T):
            a = agent.act(env.simulator.state)
            obs, r, done, _ = env.step(a)
            # env.render()
            # time.sleep(0.3)

            ret += gamma ** i * r

            if done:
                env.reset()
                print('Resetting the environment at t=%d.' % i)

        print('Horizon: %d, return: %.4f' % (horizon, ret))


if __name__ == '__main__':
    run_baselines()
