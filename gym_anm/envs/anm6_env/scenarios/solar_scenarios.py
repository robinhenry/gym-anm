import numpy as np
from calendar import isleap
from scipy.stats import norm
import datetime as dt


class SolarGenerator(object):

    def __init__(self, init_date, delta_t, np_random, p_max=1.):
        self.np_random = np_random
        self.p_max = p_max
        self.delta_t = delta_t
        self.date = init_date

        self.base_nf = 0.04
        self.nf = 0.
        self.p = 0.
        self.base = None

    def __iter__(self):
        return self

    def __next__(self):
        T_days = 365 + isleap(self.date.year)
        days_since_1jan = (self.date - dt.datetime(self.date.year, 1, 1)).days
        days_since_solstice = days_since_1jan + 10

        if self.base is None or (self.date.hour == 0 and self.date.minute == 0):
            self.base, self.sunrise, self.sunset = \
                self._next_day_base(days_since_solstice, T_days)

        timestep = int((self.date.hour * 60 + self.date.minute) / self.delta_t)

        if timestep == self.base.size - 1:
            diff = self.base[0] - self.p
        else:
            diff = self.base[timestep + 1] - self.p

        noise = self.np_random.normal(loc=diff, scale=self.nf)
        self.p += noise
        self.p = self._clip_single_production(self.p,
                                              self.date.hour + self.date.minute / 60,
                                              self.sunrise, self.sunset)

        # Increment the date.
        self.date += dt.timedelta(minutes=self.delta_t)

        return self.p * self.p_max

    def next(self):
        return self.__next__()

    def _next_day_base(self, days_since_solstice, T_days):
        sunrise, sunset = self._sunset_sunrise(days_since_solstice, T_days)
        bell = self._bell_curve(sunrise, sunset)
        scaling = self._yearly_pattern(days_since_solstice, T_days)
        base = scaling * bell

        # Add random noise.
        base += self.np_random.normal(loc=0., scale=self.base_nf, size=base.size)
        base = self._clip_base_curve(base, sunrise, sunset)

        return base, sunrise, sunset

    def _sunset_sunrise(self, days_since_solstice, T_days):
        """
        Compute the sunset and sunrise hour, based on the date of the year.
        """
        # Shift hour to be centered on December, 22nd (Winter Solstice).

        sunrise = 1.5 * np.cos(2 * np.pi * days_since_solstice / T_days) + 5.5
        sunset = 1.5 * np.cos(2 * np.pi * days_since_solstice / T_days + np.pi) \
                 + 18.5

        return sunrise, sunset

    def _bell_curve(self, sunrise, sunset):
        """
        Return the a deterministic solar generation-like (bell) curve for the day.
        """

        time = np.arange(0, 24, self.delta_t / 60)

        sigma = 2.
        y = np.exp(- (time - 12)**2 / (2 * sigma**2))
        y = self._clip_base_curve(y, sunrise, sunset)

        return y

    def _yearly_pattern(self, days_since_solstice, T_days):
        """
        Return a factor to scale solar generation, based on the day of the year.
        """
        factor = 0.25 * np.cos(2 * np.pi * days_since_solstice / T_days + np.pi) \
                 + 0.75
        return factor

    def _clip_base_curve(self, p, sunrise, sunset):
        time = np.arange(0, 24, self.delta_t / 60)

        p_clipped = np.array(p)
        p_clipped[time < sunrise] = 0
        p_clipped[time > sunset] = 0
        p_clipped[p < 0] = 0

        return p_clipped

    def _clip_single_production(self, p, h, sunrise, sunset):
        if p < 0 or h < sunrise or h > sunset:
            return 0
        else:
            return p


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(2019)
    p_max = 1
    init_date = dt.datetime(2019, 1, 1)
    delta_t = 15
    solar_generator = SolarGenerator(init_date, delta_t, rng, p_max)

    curve = []
    for i_from in range(24 * 4 * 7):
        curve.append(next(solar_generator))

    plt.plot(curve)
    plt.xlabel('Timestep (15 min)')
    plt.ylabel('Solar production')
    plt.title('Generated solar curve over a week')

    plt.show()
