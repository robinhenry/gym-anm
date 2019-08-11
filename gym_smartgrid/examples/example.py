from gym_smartgrid.envs import SmartGridEnv6
import numpy as np

if __name__ == '__main__':
    env = SmartGridEnv6()
    env.reset()

    curtailment = np.array([[10, 10], [20, 20]])
    alphas = np.array([[2], [4]])
    q_storage = np.array([[0.1], [0.2]])


    for i in range(50000):
        print(i)

        # curt = env.np_random.random(size=(2,))
        # al = env.np_random.random(size=(1,))
        # q = env.np_random.random(size=(1,))
        # env.step((curt, al, q))

        env.render(sleep_time=1)
        env.step(env.action_space.sample()) # take a random action
    # history = env.close(path='test_history.csv')
    #
    # env.replay('test_history.csv')
