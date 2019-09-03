from gym_smartgrid.envs.smartgrid_env6.anm6_hard import ANM6Hard
from gym_smartgrid.envs.smartgrid_env6.anm6_easy import ANM6Easy

if __name__ == '__main__':
    env = ANM6Easy()
    # env = SmartGridEnv6Hard()
    env.reset()
    for i in range(5):
        env.render(sleep_time=.5)
        a = env.action_space.sample()

        # Choose action.



        print(a)
        env.step(a) # select a random action


    # history = env.close(path='test_history.csv')
    # env.replay('test_history.csv')
