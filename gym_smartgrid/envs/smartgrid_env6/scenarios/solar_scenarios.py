import numpy as np
from calendar import isleap
from scipy.stats import norm


class SolarGenerator():
    def __init__(self, np_random, p_max):
        self.np_random = np_random
        self.p_max = p_max
        self.delta_t = 15
        self.prev_p = 0.
        self.prev_base = None
        self.prev_day = None

    def __iter__(self):
        return self

    def __next__(self):

        if self.prev_base is None:
            self._next_base()

        diff = self.prev_base - self.prev_p
        noise = self.np_random.normal(loc=diff, scale=self.scale)
        self.prev_p += noise

        # Clip production to 0 before sunrise and after sunset.
        if self.h < self.sunrise or self.h > self.sunset:
            self.prev_p = 0.

        # Make sure that P stays within [0, 1].
        self.prev_p = np.maximum(self.prev_p, 0.)
        self.prev_p = np.minimum(self.prev_p, 1.)

        # Compute the next base value.
        self._next_base()

        return self.prev_p * self.p_max * self.day_peak

    def next(self, date, bell_noise_factor=0.1, cloud_noise_factor=0.0,
             scale=0.005):

        if date.minute % self.delta_t:
            raise ValueError('The time should be a multiple of 15 minutes.')

        self.date = date
        self.bell_noise = bell_noise_factor
        self.cloudy_day_freq = cloud_noise_factor
        self.scale = scale

        self.T = 365 + isleap(self.date.year)
        self.h = (self.date.hour + self.date.minute / 60.) % 24.

        return self.__next__()

    def _next_base(self):
        self._sunset_sunrise_pattern()
        self.prev_base = self._bell_curve() * self._yearly_pattern()

        if self.prev_day is None or self.prev_day != self.date.day:
            self.day_peak = self._compute_day_peak()

    def _compute_day_peak(self):
        peak = self.np_random.normal(loc=0.5, scale=self.cloudy_day_freq)
        peak = np.maximum(np.minimum(peak, 1.), 0.)

        return peak

    def _sunset_sunrise_pattern(self):
        """
        Compute the sunset and sunrise hour, based on the date of the year.

        :param hour: the number of hours since January, 1st at 12:00 a.m.
        """

        self.sunset = 1.5 * np.sin(self.h * 2 * np.pi / self.T) + 18.5
        self.sunrise = 1.5 * np.sin(self.h * 2 * np.pi / self.T) + 5.5

    def _bell_curve(self):
        """
        Return the a noisy solar generation-like (bell) curve value at hour t.

        This function returns a capacity factor for solar generation following a
        bell curve (Gaussian), given a time of the day. It is assumed that hour=0
        represents 12:00 a.m., i.e. hour=26 => 2:00 a.m. Noise is also added
        sampled from a Gaussian N(0, 1).

        :param hour: hour of the day (hour=1.5 means 1:30 am).
        :param sunrise: hour of sunrise for the given day.
        :param sunset: hour of sunset for the given day.
        :return: the noisy solar generation, normalized in [0, 1].
        """

        h = (self.date.hour + self.date.minute / 60.) % 24.
        y = lambda x: norm.pdf(x, loc=12., scale=2.)
        if h > self.sunrise and h < self.sunset:
            p = y(h) / y(12.)
            # Add noise to the capacity factor (stochastic).
            p += self.bell_noise * self.np_random.normal(loc=0., scale=1.)
        else:
            p = 0.

        return p

    def _yearly_pattern(self):
        """
        Return a factor to scale solar generation, based on the time of the year.

        This function returns a factor in [0.5, 1.0] used to scale the solar
        power generation curves, based on the time of the year, following a
        simple sinusoid. The base hour=0 represents 12:00 a.m. on January,
        1st. The sinusoid is designed to return its minimum value on the
        Winter Solstice and its maximum value on the Summer one.

        :param hour: the number of hours passed January, 1st at 12:00 a.m.
        :return: a solar generation scaling factor in [0, 1].
        """

        # Shift hour to be centered on December, 22nd (Winter Solstice).
        h = self.date.hour + 240

        return 0.25 * np.sin(h * 2 * np.pi / self.T - np.pi / 2) + 0.75


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import datetime as dt

    rng = np.random.RandomState(2019)
    p_max = 1
    load_generator = SolarGenerator(rng, p_max)

    curve = []
    date = dt.datetime(2019, 1, 1)
    for i in range(24 * 4 * 3):
        curve.append(load_generator.next(date, bell_noise_factor=0.01,
                                         scale=0.01, cloud_noise_factor=0.05))
        date += dt.timedelta(minutes=15)

    plt.plot(curve)
    plt.xlabel('Timestep (15 min)')
    plt.ylabel('Consumption factor in [0, 1]')
    plt.title('Generated load curve over a week')

    plt.show()