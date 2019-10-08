import os
import pandas as pd
from calendar import monthrange
import datetime as dt
import numpy as np


class LoadGenerator(object):
    """
    This class implements the stochastic consumption process of a passive load.

    The generation of the consumption profiles is based on real historical
    demand curves, stored in memory.

    The generation of consumption profiles is done as follows:
        1.  At the beginning of each new day: select, at random, a historical
            curve from the current month. This curve will sever as the "base
            curve" for the current day.
        2.  At each time step, generate some random normal noise biased
            towards the base curve and add it to the demand from the previous
            time step. This way, the newly generated curve for the day will
            follow the pattern present in the historical curve.

    Parameters
    ----------
    folder : str
        The absolute path to the folder containing the real consumption curves.
    date : datetime.datetime
        The current time.
    delta_t : int
        The interval of time between subsequent time steps (minutes).
    np_random : numpy.random.RandomState
        The random state of the environment.
    p_max : float
        The maximum real power consumption from the load (MW). Should be < 0.
    prev_p : float
        The previous real power demand factor in [0, 1]. The amount of
        power used by the load is then prev_p * p_max (MW).
    nf : float
        The noise factor controlling the random noise added at each time step.
        E.g. nf=0 => the consumption profile is equal to the base curve.
    prev_month : int
        The month associated with the previous time step.
    prev_day : int
        The day associated with the previous time step.
    day_curve : numpy.ndarray
        The base curve of the current day.
    """

    def __init__(self, folder, init_date, delta_t, np_random, p_max=1.):
        """
        Parameters
        ----------
        folder : str
            The absolute path to the folder containing the real consumption
            curves.
        init_date : datetime.datetime
            The time corresponding to time step t=0.
        delta_t : int
            The interval of time between subsequent time steps (minutes).
        np_random : numpy.random.RandomState
            The random state of the environment.
        p_max : float, optional
            The maximum real power demand (MW). Should be < 0.
        """

        self.folder = folder
        self.np_random = np_random
        self.p_max = p_max
        self.delta_t = delta_t
        self.date = init_date

        self.nf = 0.005
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

        # Generate random noise biased towards the base curve.
        t = int((self.date.hour * 60 + self.date.minute) / self.delta_t)
        diff = self.day_curve[t - 1] - self.prev_p
        noise = self.np_random.normal(loc=diff, scale=self.nf)

        # Add random noise.
        self.prev_p += noise
        self.prev_p = np.maximum(0, np.minimum(1, self.prev_p))

        # Increment the current time.
        self.date += dt.timedelta(minutes=self.delta_t)

        return self.prev_p * self.p_max

    def next(self):
        return self.__next__()

    def _load_month_curves(self):
        """ Load in memory the base curves of the current month. """

        path = os.path.join(self.folder, f'curves_{self.date.month - 1}.csv')
        self.month_curves = pd.read_csv(path, header=None).values
        self.prev_month = self.date.month

    def _load_day_base_curve(self):
        """ Load a base curve for the new day. """

        rand_day = self.np_random.randint(0, monthrange(self.date.year,
                                                        self.date.month)[1])
        self.day_curve = self.month_curves[rand_day - 1, :]
        self.prev_day = self.date.day


if __name__ == '__main__':
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
