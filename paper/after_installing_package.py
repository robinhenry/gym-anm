import gym


if __name__ == '__main__':

    env = gym.make('gym_anm:ANM6-Easy-v0')
    env.reset()

    for i in range(10000):
        if i % 100 == 0:
            print(i)
        env.render(sleep_time=0.)
        env.step(env.action_space.sample())

    print('Done!')
