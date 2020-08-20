from gym_anm import DCOPFAgent
from run_baseline import run_baseline


def dcopf_grid_search():

    T = 3000
    seed = 1000
    savefile = './DCOPF_returns.txt'

    for planning_steps in [48, 96, 192, 288]:
        for safety_margin in [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.]:

            run_baseline(DCOPFAgent, safety_margin, planning_steps, T,
                         seed, savefile)


if __name__ == '__main__':
    dcopf_grid_search()
