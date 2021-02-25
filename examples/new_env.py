import numpy as np

from gym_anm.envs import ANM6


class NewEnvTest(ANM6):

    def __init__(self):
        observation = [('aux', [0])]
        # observation = [('dev_q', [1, 2, 3], 'pu')]
        K = 1
        delta_t = 0.25
        gamma = 0.9
        lamb = 0
        aux_bounds = None
        costs_clipping = None
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds,
                         costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series()
        self.P_maxs = _get_gen_time_series()

    def init_state(self):
        n_dev, n_gen, n_des = 7, 2, 1
        state = np.random.rand(2 * n_dev + n_des + n_gen + self.K)
        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))
        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux])
        for p_max in self.P_maxs:
            vars.append(p_max[aux])
        vars.append(aux)

        return np.array(vars)

    def reset(self, date_init=None):
        obs = super().reset()
        return obs


def _get_load_time_series():
    return np.random.rand(3, 96)


def _get_gen_time_series():
    return np.random.rand(2, 96)


if __name__ == '__main__':
    import time

    env = NewEnvTest()
    env.reset()

    T = 3
    start = time.time()
    for i in range(T):
        a = env.action_space.sample()
        o, r, _, _ = env.step(a)
        print(['%.1f' %x for x in o])
    print('Length of o_t: ', len(o))

    print('Done.')
