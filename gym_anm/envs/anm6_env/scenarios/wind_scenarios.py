import numpy as np
from calendar import isleap
import datetime as dt


class WindGenerator(object):

    def __init__(self, init_date, delta_t, np_random, p_max=1.):
        self.np_random = np_random
        self.p_max = p_max
        self.date = init_date
        self.delta_t = delta_t
        self.nf = 0.02
        self.p = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.p is None or (self.date.hour == 0 and self.date.minute == 0):
            T_days = 365 + isleap(self.date.year)
            days_since_1jan = (self.date - dt.datetime(self.date.year, 1, 1)).days
            days_since_solstice = days_since_1jan + 10

            self.p = self._yearly_pattern(days_since_solstice, T_days)

        noise = self.np_random.normal(0., scale=self.nf)
        self.p += noise

        self.p = self.p if self.p > 0. else 0.
        self.p = self.p if self.p < 1. else 1.

        # Increment the date.
        self.date += dt.timedelta(minutes=self.delta_t)

        return self.p * self.p_max

    def next(self):
        return self.__next__()

    def _yearly_pattern(self, days_since_solstice, T_days):
        """
        Return a factor to scale wind generation, based on the day of the year.
        """
        return 0.25 * np.cos(days_since_solstice * 2 * np.pi / T_days) + 0.5

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(2019)
    p_max = 1
    init_date = dt.datetime(2019, 1, 1)
    delta_t = 15
    wind_generator = WindGenerator(init_date, delta_t, rng, p_max)

    curve = []
    for i_from in range(24 * 4 * 50):
        curve.append(next(wind_generator))

    plt.plot(curve)
    plt.xlabel('Timestep (15 min)')
    plt.ylabel('Wind production')
    plt.title('Generated wind curve over a week')

    plt.show()
