import gym
from time import time


def run_baseline(agent_class, safety_margin, planning_steps, T=3000, seed=None,
                 savefile=None):

    print('Using agent: ' + agent_class.__name__)

    # Get file to write results to.
    if savefile is None:
        savefile = './Baseline_{}_results.txt'.format(agent_class.__name__)

    # Get the environment ready.
    env = gym.make('ANM6Easy-v0')
    gamma = env.gamma
    if seed is not None:
        env.seed(1000)
    env.reset()
    ret = 0.

    # Make the agent.
    agent = agent_class(env.simulator, env.action_space, gamma,
                        safety_margin=safety_margin,
                        planning_steps=planning_steps)

    start = time()
    for i in range(T):
        print(i)
        a = agent.act(env)
        obs, r, done, _ = env.step(a)

        ret += gamma ** i * r

        if done:
            env.reset()
            print('Resetting the environment at t=%d.' % i)
    elapsed = time() - start

    # Write results to file.
    with open(savefile, 'a') as f:
        f.write('T=%d, N-stage=%d, safety_margin=%.2f, return=%.3f\n'
                % (T, agent.planning_steps, agent.safety_margin, ret))

    print('Planning steps: %d, safety margin: %.2f, return: %.4f, elapsed time: %.2f sec'
          % (agent.planning_steps, agent.safety_margin, ret, elapsed))

    return ret
