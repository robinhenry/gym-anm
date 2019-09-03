import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gym_smartgrid.envs.smartgrid_env6.anm6_hard import ANM6Hard


def null_agent():
    return np.array([30, 50]), np.array([0]), np.array([0])


if __name__ == '__main__':
    env = ANM6Hard()
    env.reset()

    ts = []
    for i in range(4*24*5):
        a = null_agent()
        obs, r, _, _ = env.step(a)
        ts.append(obs[0])
    ts = np.array(ts)

    sns.set()
    # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(14, 5))
    plt.figure(figsize=(15, 5))
    axs = [plt.gca()]

    # plt.rc('text', usetex=True)
    labels = ['$P_1$', r'$\tilde P_2$', '$P_3$', r'$\tilde P_4$', '$P_5$']
    for i in range(1, 6):
        axs[0].plot(ts[:, i], label=labels[i - 1])

    axs[0].set_ylabel(r'$W(t)$ (MW)', fontsize=15)
    axs[0].legend(loc='lower left')

    bus1 = ts[:, 1] + ts[:, 2]
    bus2 = ts[:, 3] + ts[:, 4]
    bus3 = ts[:, 5]
    buses = [bus1, bus2, bus3]

    # for i in range(3):
    #     axs[1].plot(buses[i], label=rf'$P_{i+1}$')
    #
    # axs[1].set_xlabel('Time steps of 15 minutes', fontsize=15)
    # axs[1].set_ylabel('P (MW)', fontsize=15)
    # axs[1].legend()
    # axs[1].legend(loc='lower left')

    axs[0].set_xlabel(rf'Time step $t$ (15 minutes)', fontsize=15)

    plt.tight_layout()
    plt.savefig('hard_scenario.png', tight_layout=True)

    plt.show()