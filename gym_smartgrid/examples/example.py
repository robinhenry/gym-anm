from gym_smartgrid.envs import SmartGridEnv6
import numpy as np

if __name__ == '__main__':
    env = SmartGridEnv6()
    env.reset()

    for i in range(50000):
        print(i)

        env.render(sleep_time=.01)
        #
        # curt = env.np_random.random(size=(2,))
        # al = env.np_random.random(size=(1,))
        # q = env.np_random.random(size=(1,))
        # env.step((curt, al, q))

        env.render(sleep_time=.01)
        env.step(env.action_space.sample()) # take a random action
    # history = env.close(path='test_history.csv')
    #
    # env.replay('test_history.csv')
