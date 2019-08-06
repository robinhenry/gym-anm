import numpy as np
import os

from .stochastic_processes import VRESet, LoadSet


def init_vre(wind, solar, delta_t, noise_factor=0.1, np_random=None):
    """
    Return generator objects to model wind and solar generation.

    This function returns a python generator object for each generator device
    (wind or solar), modelling a stochastic process.

    :param wind: a dict with {dev_idx: P_max} for wind power generation devices.
    :param solar: a dict with {dev_idx: P_max} for solar power devices.
    :param delta_t: 0.25 if each time step has a 15-min length.
    :param noise_factor: a factor multiplying noise sampled from N(0, 1).
    :return: a list of generator objects, one for each generator.
    """

    return VRESet(wind, solar, delta_t, noise_factor, np_random)

def init_load(factors, delta_t, np_random):

    # Load basic demand curves from files.
    curves_per_month = _load_demand_curves()

    load_generators = LoadSet(delta_t, factors, curves_per_month, np_random)

    return load_generators

def _load_demand_curves():
    """
    Load and return demand curves stored in .csv files.

    This function loads the data stored in folder 'data_demand_curves' as a
    list of ndarray, one for each month of the year (in order). Each array is a
    N_day x 96 matrix, where element [i, j] of the array represents the load
    demand on day i, at timestep j, where each timestep is assumed to last 15
    minutes. The data is normalized to be in [0.2, 0.8].

    :return: a list of 12 ndarray, each containing 29-31 demand curves.
    """

    curves_per_month = []
    for i in range(12):
        current_file = os.path.dirname(__file__)
        filename = os.path.join(current_file,
                                'data_demand_curves',
                               'curves_' +  str(i) + '.csv')
        curves = np.loadtxt(filename, delimiter=',')
        curves_per_month.append(curves)

    return curves_per_month


if __name__ == '__main__':
    pass
