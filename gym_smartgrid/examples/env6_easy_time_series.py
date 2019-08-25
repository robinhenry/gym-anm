import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from gym_smartgrid.envs import SmartGrid6Easy


def null_agent():
    return np.array([30, 50]), np.array([0]), np.array([0])

def draw_regions(axs):
    a = 0.15
    for ax in axs:
        ax.axvspan(0, 23, alpha=a, color='red')
        ax.axvspan(23, 31, alpha=a, color='grey')
        ax.axvspan(31, 43, alpha=a, color='green')
        ax.axvspan(43, 51, alpha=a, color='grey')
        ax.axvspan(51, 63, alpha=a, color='blue')
        ax.axvspan(63, 71, alpha=a, color='grey')
        ax.axvspan(71, 83, alpha=a, color='green')
        ax.axvspan(83, 91, alpha=a, color='grey')
        ax.axvspan(91, 95, alpha=a, color='red')

        ax.set_xlim([0, 94])

        y = 20
        fontsize = 20
        ax.text(11, y, '1', fontsize=fontsize)
        ax.text(36, y, '2', fontsize=fontsize)
        ax.text(56, y, '3', fontsize=fontsize)
        ax.text(76, y, '2', fontsize=fontsize)
        ax.text(92, y, '1', fontsize=fontsize)



if __name__ == '__main__':
    env = SmartGrid6Easy()
    env.reset()

    ts = []
    for i in range(4*24):
        a = null_agent()
        obs, r, _, _ = env.step(a)
        ts.append(obs[0])
    ts = np.array(ts)

    sns.set()
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(14, 5))

    # plt.rc('text', usetex=True)
    labels = ['$P_1$', r'$\tilde P_2$', '$P_3$', r'$\tilde P_4$', '$P_5$']
    for i in range(1, 6):
        axs[0].plot(ts[:, i], label=labels[i - 1])

    axs[0].set_ylabel('P (MW)', fontsize=15)
    axs[0].legend(loc='lower left')

    bus1 = ts[:, 1] + ts[:, 2]
    bus2 = ts[:, 3] + ts[:, 4]
    bus3 = ts[:, 5]
    buses = [bus1, bus2, bus3]

    for i in range(3):
        axs[1].plot(buses[i], label=rf'$P_{i+1}$')

    axs[1].set_xlabel('Time steps of 15 minutes', fontsize=15)
    axs[1].set_ylabel('P (MW)', fontsize=15)
    axs[1].legend()
    axs[1].legend(loc='lower left')

    draw_regions(axs)

    plt.tight_layout()
    plt.savefig('easy_scenario.png', tight_layout=True)

    plt.show()