import gym
from gym_anm import MPCAgent


def run():
    env = gym.make('ANM6Easy-v0')
    o = env.reset()

    # Initialize the MPC policy.
    agent = MPCAgent(env.simulator, env.action_space, env.gamma,
                     safety_margin=0.96, planning_steps=10)

    # Run the policy.
    for i in range(100):
        a = agent.act(env)
        obs, r, done, _ = env.step(a)


if __name__ == '__main__':
    run()