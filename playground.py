# from gym_anm.envs import ANM6Easy
# import webbrowser
# from tqdm import tqdm


def test_sever():
    p = 'http://127.0.0.1:8000/envs/anm6_env/rendering/'
    webbrowser.open_new_tab(p)


def test_dependencies():
    import gym
    # import gym_anm

    env = gym.make('gym_anm:ANM6-Easy-v0')
    env.reset()

    print('Environment reset and ready.')

    T = 50
    for i in range(T):
        print(i)
        a = env.action_space.sample()
        o, r, _, _ = env.step(a)

    print('Done with {} steps!'.format(T))


def run():
    env = ANM6Easy()
    o = env.reset()

    T = int(1e4)
    for i in tqdm(range(T)):
        a = env.action_space.sample()
        o, r, done, info = env.step(a)
        # env.render(mode='human', sleep_time=2)
    print('Done with %d timesteps!' % T)


if __name__ == '__main__':
    # run()
    # test_sever()
    test_dependencies()
