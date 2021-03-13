"""
This script illustrates how to interact with gym-anm environments. In this example, the agent samples random
actions from the action space of the ANM6Easy-v0 task for 1000 timesteps. Every time a terminal state is reached, the
environment gets reset.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/using_env.html.
"""
import gym
import time

def run():
    env = gym.make('gym_anm:ANM6Easy-v0')
    o = env.reset()

    for i in range(10):
        a = env.action_space.sample()
        o, r, done, info = env.step(a)
        env.render()
        time.sleep(0.5)   # otherwise the rendering is too fast for the human eye

        if done:
            o = env.reset()
    env.close()

if __name__ == '__main__':
    run()