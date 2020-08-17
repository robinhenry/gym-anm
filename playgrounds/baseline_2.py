import time
import os

from gym_anm.envs import ANM6Easy
from gym_anm import DCOPFAgent, MPCAgentANM6Easy


def run_baselines(agent_class):
    print('Using agent: ' + agent_class.__name__)

    env = ANM6Easy()
    gamma = env.gamma

    # Parameters.
    T = int(3000)
    horizons = [1, 4, 2*4, 6*4, 12*4, 24*4, 24*4*2, 24*4*5]
    safety_margins = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    savefile = './MPC_ANM6Easy_results.txt'

    for horizon in horizons:
        for sm in safety_margins:
            env.seed(1000)
            env.reset()

            agent = agent_class(env.simulator, env.action_space, gamma,
                                planning_steps=horizon, safety_margin=sm)
            ret = 0.
            for i in range(T):
                a = agent.act(env)
                obs, r, done, _ = env.step(a)
                ret += gamma ** i * r

                if done:
                    env.reset()
                    print('Resetting the environment at t=%d.' % i)

            # Write results to file.
            with open(savefile, 'a') as f:
                f.write('T=%d, N-stage=%d, safety_margin=%.2f, return=%.3f\n'
                        % (T, horizon, sm, ret))

            print('Horizon: %d, safety margin: %.2f, return: %.4f' % (horizon, sm, ret))


if __name__ == '__main__':
    # run_baselines(DCOPFAgent)

    run_baselines(MPCAgentANM6Easy)
