from gym_anm.envs.anm6_env.anm6 import ANM6

class ANM6Easy(ANM6):
    def __init__(self):
        super().__init__()

    def init_load(self, load_pmax, init_date, delta_t, np_random):
        scenario1 = [-2, -20, 0]
        scenario2 = [-5, -10, -25]
        scenario3 = [-1, -4, 0]

        loads = []
        for i in range(len(scenario1)):
            loads.append(self._constant_generator(scenario1[i], scenario2[i],
                                                  scenario3[i]))
        return loads

    def init_dg(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
        scenario1 = [30, 40]
        scenario2 = [4, 11]
        scenario3 = [0, 40]

        vres = []
        for i in range(len(scenario1)):
            vres.append(self._constant_generator(scenario1[i], scenario2[i],
                                                  scenario3[i]))
        return vres

    def init_soc(self, soc_max):
        return [s / 2. for s in soc_max]

    def _constant_generator(self, case1, case2, case3):
        transition_time = 8

        while True:

            # Scenario 3.
            if self.time.hour < 6:
                yield case3

            # Transition 3 -> 2.
            elif 6 <= self.time.hour < 8:
                for t in range(1, transition_time + 1):
                    diff = case3 - case2
                    yield (case3 - t * diff / transition_time)

            # Scenario 2.
            elif 8 <= self.time.hour < 11:
                yield case2

            # Transition 2 -> 1.
            elif 11 <= self.time.hour < 13:
                for t in range(1, transition_time + 1):
                    diff = case2 - case1
                    yield (case2 - t * diff / transition_time)

            # Scenario 1.
            elif 13 <= self.time.hour < 16:
                yield case1

            # Transition 1 -> 2.
            elif 16 <= self.time.hour < 18:
                for t in range(1, transition_time + 1):
                    diff = case1 - case2
                    yield (case1 - t * diff / transition_time)

            # Scenario 2.
            elif 18 <= self.time.hour < 21:
                yield case2

            # Transition 2 -> 3.
            elif 21 <= self.time.hour < 23:
                for t in range(1, transition_time + 1):
                    diff = case2 - case3
                    yield (case2 - t * diff / transition_time)

            # Scenario 3.
            else:
                yield case3