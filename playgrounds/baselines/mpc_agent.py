from gym_anm import MPCAgentANM6Easy
from run_baseline import run_baseline


def mpc_grid_search():

    T = 3000
    seed = 1000
    savefile = './MPC_returns.txt'

    for planning_steps in [288, 384, 480]:
        for safety_margin in [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.]:

            run_baseline(MPCAgentANM6Easy, safety_margin, planning_steps, T,
                         seed, savefile)


if __name__ == '__main__':
    mpc_grid_search()
