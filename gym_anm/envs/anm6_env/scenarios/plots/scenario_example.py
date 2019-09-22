import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from gym_anm.envs.anm6_env.scenarios.solar_scenarios import SolarGenerator
from gym_anm.envs.anm6_env.scenarios.wind_scenarios import WindGenerator
from gym_anm.envs.anm6_env.scenarios.load_scenarios import LoadGenerator


def run():
    rng = np.random.RandomState(2019)
    delta_t = 15
    t_length = 24 * 4 * 5

    dates = [dt.datetime(2019, 1, 1), dt.datetime(2019, 5, 1),
             dt.datetime(2019, 9, 1)]
    p_max = [10, 30, 30, 50, 30]
    folder_house = '../data_demand_curves/house'
    folder_factory = '../data_demand_curves/factory'

    for i_date, date in enumerate(dates):
        house = LoadGenerator(folder_house, date, delta_t, rng, p_max[0])
        solar = SolarGenerator(date, delta_t, rng, p_max[1])
        industry = LoadGenerator(folder_factory, date, delta_t, rng, p_max[2])
        wind = WindGenerator(date, delta_t, rng, p_max[3])
        ev = LoadGenerator(folder_house, date, delta_t, rng, p_max[4])

        p_house, p_pv, p_industry, p_wind, p_ev = [], [], [], [], []
        for i in range(t_length):
            p_house.append(next(house))
            p_pv.append(next(solar))
            p_industry.append(next(industry))
            p_wind.append(next(wind))
            p_ev.append(next(ev))

        f, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 7))

        ax0.plot(p_house, label='$-P_1$')
        ax0.plot(p_pv, label=r'$\tilde  P_2$')
        ax0.plot(p_industry, label='$-P_3$')
        ax0.plot(p_wind, label=r'$\tilde P_4$')
        ax0.plot(p_ev, label='$-P_5$')
        ax0.set_ylim([0, 50])
        ax0.set_ylabel('$|P|$', fontsize=15)
        ax0.legend(loc='upper right')

        load = np.sum(np.array([p_house, p_industry, p_ev]), axis=0)
        gen = np.sum(np.array([p_pv, p_wind]), axis=0)
        ax1.plot(load, label='Total real demand', color='#4960fb')
        ax1.plot(gen, label='Total real generation', color='#31b620')
        ax1.set_ylim([0, 70])
        ax1.set_ylabel('$|P|$', fontsize=15)
        ax1.legend(loc='lower right')

        plt.xlim([0, t_length])
        plt.xlabel('Time step $t$ (15 minutes)', fontsize=15)
        plt.show()

        f.savefig(f'anm6_examples_{i_date}', tight_layout=True)

if __name__ == '__main__':
    run()