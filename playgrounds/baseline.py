import time

from gym_anm.envs import ANM6Easy
from gym_anm import DCOPFAgent, MPCAgentANM6Easy


def run_baselines(agent_class):
    env = ANM6Easy()
    gamma = env.gamma
    T = int(1000)

    print('Using agent: ' + agent_class.__name__)

    horizon = 1

    for sm in [0.8, 0.85, 0.9, 0.93, 0.96, 0.99, 1]:
        env.seed(1000)
        env.reset()

        agent = agent_class(env.simulator, env.action_space, gamma,
                            planning_steps=horizon, safety_margin=sm,
                            des_eff=[0.9])
        ret = 0.

        for i in range(T):
            a = agent.act(env)
            obs, r, done, _ = env.step(a)
            # env.render()
            # time.sleep(0.3)
            # print('i = ', i, ', r = ', r)

            ret += gamma ** i * r

            if done:
                env.reset()
                print('Resetting the environment at t=%d.' % i)

        print('Horizon: %d, safety margin: %.2f, return: %.4f' % (horizon, sm, ret))


if __name__ == '__main__':
    # run_baselines(DCOPFAgent)

    run_baselines(MPCAgentANM6Easy)
