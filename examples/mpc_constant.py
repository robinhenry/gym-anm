"""
This script shows how to run the MPC-based DC OPF policy
:math:`\\pi_{MPC-N}^{constant}` in an arbitrary gym-anm
environment.

This policy assumes constant demand and generation during the
optimization horizon.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#constant-forecast.
"""
import gymnasium as gym
from gym_anm import MPCAgentConstant


def run():
    env = gym.make("ANM6Easy-v0")
    o, _ = env.reset()

    # Initialize the MPC policy.
    raw_env = env.unwrapped
    agent = MPCAgentConstant(raw_env.simulator, env.action_space, raw_env.gamma, safety_margin=0.96, planning_steps=10)

    # Run the policy.
    for t in range(100):
        a = agent.act(env.unwrapped)
        obs, r, terminated, _, _ = env.step(a)
        print(f"t={t}, r_t={r:.3}")


if __name__ == "__main__":
    run()
