.. gym-anm tutorial documentation

.. _anm6_label:

ANM6-Easy
=============
The simplest :code:`gym-anm` task is :code:`ANM6Easy-v0`, which was specifically designed as a toy example that
highlights common challenges faced in ANM of distribution networks.

*Note*: although the task is simple compared to real-world scenarios, training RL agents to perform well on it is not
straightforward. **We highly recommend you start with this one before moving on to more challenging ANM tasks.**

Environment overview
---------------------
The characteristics of :code:`ANM6Easy-v0` are:

* The distribution network contains 6 buses, with one high-voltage to low-voltage transformer (see screenshots below).
* There are 3 passive loads (1 industrial complex, 1 residential area, and 1 EV charging garage), 2 renewable
  energy generators (1 wind farm and 1 PV plant), 1 storage unit, and 1 fossil fuel generator (the slack bus).
* A time discretization of :math:`\Delta t = 0.25` (i.e., 15 minutes) is used by analogy with the typical duration of a
  market period.
* The environment is fully observable, with :math:`o_t = s_t`.
* The discount factor is fixed to :math:`\gamma = 0.995` and the reward penalty hyperparameter to :math:`\lambda = 10^3`.

In order to limit the complexity of the task, the environment was designed to be fully deterministic:

* Both the demand from loads and the maximum generation (before curtailment) profiles from the renewable energies are
  modelled as fixed 24-hour time series that repeat every day, indefinitely.
* This is achieved by using a single auxiliary variable :math:`aux_t^{(0)}` that represents the time of day and that can
  be used to index the fixed 24-hour time series. Note that the presence of :math:`aux_t^{(0)}` makes the task an MDP,
  since future loads and generations can be expressed as functions of the time of day.
* These 24-hour time series were divided into 3 particular scenarios, each designed to highlight a specific challenge
  in ANM.

Scenarios
---------
Each 24-hour day is divided into 4 periods (note that scenario 2 happens twice each day):

* Period 1: scenario 1, from 23:00 to 06:00,
* Period 2: scenario 2, from 08:00 to 11:00,
* Period 3: scenario 3, from 13:00 to 16:00,
* Period 4: scenario 2, from 18:00 to 21:00.

During each period (scenario), the different power injections remain constant. Subsequent periods are separated by
2-hour-long transitions, during which the power injections linearly increase/decrease from their old to new value.

The three subsections below show the power flows and voltage levels in the network that would result if the agent did
nothing, i.e., if the agent always chooses action vectors :math:`a_t = [30, 50, 0, 0, 0, 0]`. This is equivalent to:

* the renewable energy generations are not curtailed,
* the renewable energy generator reactive power set-points are set to 0 (:math:`Q=0`)
* the storage unit is not used (:math:`P=Q=0`).


Scenario 1
^^^^^^^^^^
This situation characterizes a windy night, when the consumption is low, the PV production null, and the wind
production at its near maximum. Due to the very low demand from the industrial load, the wind production must be
curtailed to avoid an overheating of the transmission lines connecting buses 0 and 4. This is also a period during
which the agent might use this extra generation to charge the storage unit in order to prepare to meet the large morning
demand from the EV charging garage (see situation 2).

.. image:: ../images/situation_1.*
    :width: 800

Scenario 2
^^^^^^^^^^
In this situation, bus 5 is experiencing a substantial demand due to a large number of EVs being plugged-in at around
the same time. This could happen in a large public EV charging garage. In the morning, workers of close-by companies
would plug-in their car after arriving at work and, in the evening, residents of the area would plug-in their cars
after getting home. In order to emphasize the problems arising from this large localized demand, we assume that the
other buses (3 and 4) inject or withdraw very little power into/from the network, i.e. that the generation and
production at these buses roughly cancel each other out. During those periods of the day, the storage unit must provide
enough power to insure that the transmission path from bus 0 to bus 5 is not over-rated, which would lead to an
overheating of the line. For this to be possible, the agent must strategically plan ahead to ensure a sufficient
charge level at the storage unit.

.. image:: ../images/situation_2.*
    :width: 800

Scenario 3
^^^^^^^^^^
Situation 3 represents a scenario that might occur in the middle of a sunny windy weekday, during which no one is
home to consume the solar energy produced by residential PVs at bus 1 and the wind energy production exceeds the
industrial demand at bus 2. In this case, both renewable generators should be adequately curtailed while again storing
some of the extra energy to anticipate the EV late afternoon charging period of situation 2.

.. image:: ../images/situation_3.*
    :width: 800
