import numpy as np

from gym_smartgrid.envs import SmartGrid6Easy


def agent(time):
    solar_max = 30
    wind_max = 50

    wind = [25, wind_max, 18]
    solar = [18, solar_max, solar_max]
    p_su = [17, -20, 15]
    q_su = [0, 0, 0]

    curt, p, q = None, None, None

    # Scenario 3.
    if time.hour < 6:
        curt = [solar[2], wind[2]]
        p = p_su[2]
        q = q_su[2]

    # Transition 3 -> 2.
    elif 6 <= time.hour < 8:
        curt = [solar[2], wind[2]]
        p = -10
        q = 0

    # Scenario 2.
    elif 8 <= time.hour < 11:
        curt = [solar[1], wind[1]]
        p = p_su[1]
        q = q_su[1]

    # Transition 2 -> 1.
    elif 11 <= time.hour < 13:
        for t in range(1, 5):
            curt = [15, wind[1]]
            p = -5
            q = q_su[1]

    # Scenario 1.
    elif 13 <= time.hour < 16:
        curt = [solar[0], wind[0]]
        p = p_su[0]
        q = q_su[0]

    # Transition 1 -> 2.
    elif 16 <= time.hour < 18:
        curt = [solar[0], wind[0]]
        p = - 8
        q = q_su[0]

    # Scenario 2.
    elif 18 <= time.hour < 21:
        curt = [solar[1], wind[1]]
        p = - 10
        q = q_su[1]

    # Transition 2 -> 3.
    elif 21 <= time.hour < 23:
        curt = [solar[1], wind[2]]
        p = - 5
        q = q_su[1]

    # Scenario 3.
    else:
        curt = [solar[2], wind[2]]
        p = p_su[2]
        q = q_su[2]

    return np.array(curt), np.array([p]), np.array([q])

def null_agent():
    return np.array([30, 50]), np.array([0]), np.array([0])


if __name__ == '__main__':
    env = SmartGrid6Easy()
    obs = env.reset()

    for i in range(1000):
        env.render(sleep_time=.5)

        # a = null_agent()
        a = agent(env.time)
        obs, r, done, info = env.step(a)
        print('Reward: ', r)

    env.close()
