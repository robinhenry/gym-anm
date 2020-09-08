import numpy as np

from .anm6 import ANM6
from .anm6_easy import ANM6Easy, _get_load_time_series, _get_gen_time_series


class ANM6Partial(ANM6Easy):

    def __init__(self):

        observation = [
            ('branch_i_magn', 'all', 'pu'),
            # ('branch_i_ang', 'all', 'rad'),  # Not available
            ('bus_v_magn', [0, 1, 2, 3, 4, 5], 'kV'),
            ('bus_v_ang', [0, 1, 2, 3, 4, 5], 'rad'),
            ('des_soc', [6], 'MWh'),
            ('aux', [0], None),  # Key error in
        ]

        K = 1
        delta_t = 0.25         # 15 minutes between timesteps
        gamma = 0.995
        lamb = 100
        aux_bounds = np.array([[0, 24 / delta_t - 1]])
        costs_clipping = (1, 100)
        ANM6.__init__(self, observation, K, delta_t, gamma, lamb,
                      aux_bounds, costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series()
        self.P_maxs = _get_gen_time_series()


if __name__ == '__main__':
    import time

    env = ANM6Partial()
    env.reset()
    print('Environment reset and ready.')

    T = 50
    start = time.time()
    for i in range(T):
        print(i)
        a = env.action_space.sample()
        o, r, _, _ = env.step(a)

    print('Done with {} steps in {} seconds!'.format(T, time.time() - start))
