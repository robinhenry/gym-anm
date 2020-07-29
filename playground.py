from gym_anm.envs import ANM6Easy
import webbrowser
from tqdm import tqdm


def test_sever():
    p = 'http://127.0.0.1:8000/envs/anm6_env/rendering/'
    webbrowser.open_new_tab(p)


def test_runtime():
    import time

    env = ANM6Easy()
    env.reset()

    T = int(1e2)
    start = time.time()
    for i in tqdm(range(T)):
        a = env.action_space.sample()
        o, r, _, _ = env.step(a)

    print('Done with {} steps!'.format(T))
    print('Average time per step is %.3f seconds.' % ((time.time() - start) / T))


if __name__ == '__main__':
    # run()
    # test_sever()
    test_runtime()
