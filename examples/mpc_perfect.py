"""
This script shows how to run the MPC-based DC OPF policy
:math:`\\pi_{MPC-N}^{perfect}` in the ANM6Easy-v0 environment.

This policy assumes perfect forecasts of demands and generations
over the optimization horizon.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#perfect-forecast.
"""
import gym
from gym_anm import MPCAgentPerfect

def run():
    env = gym.make('ANM6Easy-v0')
    o = env.reset()

    # Initialize the MPC policy.
    agent = MPCAgentPerfect(env.simulator, env.action_space, env.gamma,
                            safety_margin=0.96, planning_steps=10)

    # Run the policy.
    for t in range(100):
        a = agent.act(env)
        obs, r, done, _ = env.step(a)
        print(f't={t}, r_t={r:.3}')

if __name__ == '__main__':
    run()