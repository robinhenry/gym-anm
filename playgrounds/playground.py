from gym_anm.envs import ANM6Easy
import time
import numpy as np
from gym_anm.simulator.solve_load_flow import solve_pfe_newton_raphson


def test_runtime():
    env = ANM6Easy()
    env.reset()

    rs = []
    e_losses, penalties = [], []

    T = int(1e4)
    start = time.time()
    for i in range(T):
        a = env.action_space.sample()
        o, r, _, _ = env.step(a)
        env.render(skip_frames=0)
        time.sleep(.5)
        if env.pfe_converged:
            rs.append(r)
            e_losses.append(env.e_loss)
            penalties.append(env.penalty)

        if i % 100 == 0:
            print('Maximum reward: %.3f' % np.max(np.abs(rs)))
            print('Maximum energy loss cost %.3f' % np.max(e_losses))
            print('Maximum penalty %.3f' % np.max(penalties))

    env.close()

    print('')
    print('Done with {} steps!'.format(T))
    print('Average time per step is %.3f seconds.' % ((time.time() - start) / T))
    print('Maximum reward: %.3f' % np.max(rs))
    print('Maximum energy loss cost %.3f' % np.max(e_losses))
    print('Maximum penalty %.3f' % np.max(penalties))


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
    # test_runtime()
    test_limits()
