from gym_anm.envs import ANM6Easy
import webbrowser
from tqdm import tqdm
from gym_anm.simulator.solve_load_flow import solve_pfe_newton_raphson


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
        env.render(sleep_time=0.)

    env.close()

    print('Done with {} steps!'.format(T))
    print('Average time per step is %.3f seconds.' % ((time.time() - start) / T))


def test_limits():
    env = ANM6Easy()
    env.reset()
    env.render()

    # Make all device inject maximum power.
    for i in [1, 3, 5]:
        # env.simulator.devices[i].p = 0.
        # env.simulator.devices[i].q = 0.
        pass
    for i in [2, 4, 6]:
        env.simulator.devices[i].p = env.simulator.devices[i].p_max
        env.simulator.devices[i].q = -env.simulator.devices[i].q_max
    env.simulator._get_bus_total_injections()
    _, env.simulator.pfe_converged = solve_pfe_newton_raphson(env.simulator)
    env.simulator.state = env.simulator._gather_state()
    env.render()

    print('')

if __name__ == '__main__':
    # test_sever()
    # test_runtime()
    test_limits()