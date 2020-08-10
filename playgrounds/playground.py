from gym_anm.envs import ANM6Easy
import time
import numpy as np
from gym_anm.simulator.solve_load_flow import solve_pfe_newton_raphson


def test_runtime():
    env = ANM6Easy()
    env.reset()

    T = int(1e4)
    start = time.time()
    done = False
    for i in range(T):

        if done:
            env.reset()
            print('Resetting at i=%d' % i)

        a = env.action_space.sample()
        o, r, done, _ = env.step(a)
        env.render(skip_frames=0)
        time.sleep(0)

        print('r= = %.4f' % r)

    env.close()

    print('')
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
    test_runtime()
    # test_limits()
