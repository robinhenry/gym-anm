import os
import pandas as pd
from calendar import monthrange
import datetime as dt


class LoadGenerator(object):

    def __init__(self, folder, init_date, delta_t, np_random, p_max=1.):
        self.folder = folder
        self.np_random = np_random
        self.p_max = p_max
        self.delta_t = delta_t
        self.date = init_date

        self.scale = 0.005
        self.prev_month = None
        self.prev_day = None
        self.prev_p = None

    def __iter__(self):
        return self

    def __next__(self):

        # A new month is reached.
        if self.date.month != self.prev_month or self.prev_month is None:
            self._load_month_curves()

        # A new day is reached.
        if self.date.day != self.prev_day or self.prev_day is None:
            self._load_day_base_curve()

        # Initialize the previous injection at the first time step.
        if self.prev_p is None:
            self.prev_p = self.day_curve[0]

        t = int((self.date.hour * 60 + self.date.minute) / self.delta_t)
        diff = self.day_curve[t - 1] - self.prev_p
        noise = self.np_random.normal(loc=diff, scale=self.scale)

        self.prev_p += noise

        # Increment the current time.
        self.date += dt.timedelta(minutes=self.delta_t)

        return self.prev_p * self.p_max

    def next(self):
        return self.__next__()

    def _load_month_curves(self):
        path = os.path.join(self.folder, f'curves_{self.date.month - 1}.csv')
        self.month_curves = pd.read_csv(path, header=None).values
        self.prev_month = self.date.month

    def _load_day_base_curve(self):
        rand_day = self.np_random.randint(0, monthrange(self.date.year,
                                                        self.date.month)[1])
        self.day_curve = self.month_curves[rand_day - 1, :]
        self.prev_day = self.date.day


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    folder_house = 'data_demand_curves/house'
    folder_factory = 'data_demand_curves/factory'
    rng = np.random.RandomState(2019)
    date = dt.datetime(2019, 1, 1)

    p_max = 1
    house = LoadGenerator(folder_house, date, 15, rng, p_max)
    factory = LoadGenerator(folder_factory, date, 15, rng, p_max)

    for gen in [house, factory]:
        curve = []
        for i_from in range(24 * 4):
            curve.append(next(gen))

        plt.plot(curve)
        plt.xlabel('Timestep (15 min)')
        plt.ylabel('Consumption factor in [0, 1]')
        plt.title('Generated load curve over a week')
        plt.show()
