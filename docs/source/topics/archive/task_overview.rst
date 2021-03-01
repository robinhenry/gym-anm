..

.. _task_overview_label:

Task overview
=============

Each :code:`gym-anm` task can be described as a `Markov Decision Process <https://en.wikipedia.org/wiki/Markov_decision_process>`_ (MDP),
of which an overview is provided below.

Goal
----
When tackling tasks modelled by :code:`gym-anm`, the goal is to minimize the cost of operating the distribution network
while avoiding the violation of grid constraints. In doing so, the agent takes the place of the Distribution Network
Operator (DNO).

Because real-world operating costs come from a wide range of sources (e.g., electricity market price, equipment
maintenance, etc.), the true operating cost must be approximated to be practical. In :code:`gym-anm`, the operating
cost is assumed to be fully described by a combination of:

1. *Energy losses*: from cable transmission (dissipated as heat) and renewable energy curtailment (clamping the output
   of a generator that could otherwise have produced more).
2. *Network constraint violation*: violating network constraints (e.g., transmission line current constraints) may lead
   to damaging parts of the power grid. In general, these costs are more important than energy losses. Note that, in
   the worst case, failing to satisfy network constraints can lead to some form of network collapse (e.g., blackout).

In addition, we restrict network constraints to two types of constraints. These constraints alone already represent
most practical limits DNOs face in the real-world management of distribution networks:

1. *Voltage constraints*: the voltage of each node of the network must remain within a specified range (e.g., [0.95, 1.05] pu).
   This is required to ensure the power grid remains stable and that all devices connected to it can operate properly.
2. *Branch current constraints*: transmission line currents must remain below certain pre-specified value (known as the
   *rated* value) to prevent lines and transformers from overheating.


Reward signal
-------------
In order to drive the behavior of RL agents towards the goal mentioned above, :code:`gym-anm` uses a reward function
that directly incorporates the quantities to be minimized: energy losses and network constraint violations.

The reward signal :math:`r_t = r(s_t, a_t, s_{t+1})` is computed as:

.. math::
    \begin{align}
        r_t =
        \begin{cases}
            -(\Delta E_{t:t+1} + \lambda \phi(s_{t+1})), & \text{if } s_{t+1} \notin \mathcal S^{terminal}, \\
            - \frac{r^{max}}{1 - \gamma}, & \text{if } s \notin \mathcal S^{terminal} \text{ and }  s_{t+1} \in \mathcal S^{terminal}, \\
            0, & \text{else,}
        \end{cases}
    \end{align}

where:

* :math:`\Delta E_{t:t+1}` is the total network-wide energy loss during :math:`(t,t+1]`,
* :math:`\phi(s_{t+1})` is a penalty term associated with the violation of operating constraints,
* :math:`\lambda` is a weighting hyperparameter,
* :math:`r^{max}` is an upper bound on the rewards emitted by the environment, often chosen around 100, and
* :math:`\gamma \in [0, 1]` is the discount factor.

During the transition from a nonterminal state to a terminal one (i.e., when the network collapses), the environment
emits a large negative reward and subsequent rewards are always zero, until a new trajectory is started by sampling a
new initial state :math:`s_0`.

For more information about the rewards, see :ref:`rewards_label`.


Action vectors
--------------
At each timestep, the agent must choose a set of actions to perform in the environment. These correspond to the
management strategy employed by the DNO.

The actions are collected into an action vector :math:`a_t \in \mathcal A`. In all :code:`gym-anm` tasks, each action
vector contains 4 types of decision variables:

* An upper limit on the active power that each generator can produce. In the case of renewable energy resources, this
  corresponds to the curtailment value. For classical generators, it corresponds to a set-point specified by the DNO.
* A set-point for the reactive power generation of each generator (renewable or not).
* A set-point for the active power injection from each energy storage unit.
* A set-point for the reactive power injection from each energy storage unit.

The resulting action space is usually constrained. Whenever the agent selects an action that falls outside of the allowed
action space, the selected action is first mapped to the physically-possible action before being applied in the environment.

For more information about action vectors, see :ref:`action_space_label`.


State vectors
-------------
Because :code:`gym-anm` tasks are modelled as MDPs, environments can always be described by their current Markovian
state, which we denote :math:`s_t \in \mathcal S`.

In all :code:`gym-anm` environments, state vectors contain the following information:

* The current (instantaneous) amount of power injected into (or withdrawn from) the power grid by each electrical device connected to it.
* The current SoC of all energy storage units (e.g., batteries).
* The maximum (theoretical) generation that each renewable energy resource could have produced if not curtailed, given
  the current environmental conditions.
* Any additional variables required to make the task Markovian (i.e., ensure that :math:`s_{t+1}` can be expressed
  probabilistically given :math:`s_t` and :math:`a_t`). We refer to these as *auxiliary variables*.

The environment may also end up in a terminal state :math:`s \in \mathcal S^{terminal} \subset \mathcal S`, which marks
the end of the episode. Reaching a terminal state indicates that the power grid has collapsed, often due to a `voltage
collapse problem <https://www.igi-global.com/dictionary/voltage-collapse/63464>`_. The environment will remain in a
terminal state until it is reset.

For more information about state vectors, see :ref:`state_space_label`.


State transitions
-----------------
Each state transition from :math:`s_t` to :math:`s_{t+1}` are fully handled by the environment. They occur in three steps:

1. A new outcome for the stochastic processes modelled by the environment is sampled. These include (a) the demand from
   each load device, (b) the maximum generation from each generator, and (c) the auxiliary variables.
2. Once the action :math:`a_t \in \mathcal A` has been selected by the agent, the action vector is mapped onto the set
   of physically possible actions :math:`A(s_t)`.
3. The mapped actions are then applied in the environment and the new electrical quantities are computed, resulting in
   a new state :math:`s_{t+1}`, observation :math:`o_{t+1}`, and reward :math:`r_t`.

.. For more information about state transitions, see :ref:`transition_label`.

Observation vectors
-------------------
In general, DNOs rarely have access to the full state of the distribution network when doing ANM.

One of the key characteristics of :code:`gym-anm` is that new environments built using this framework allows users to
easily define their own observation vectors. This means that the same task can be rendered more or less difficult by
simply modifying the observation space, thus restricting the amount (or quality) of the information the agent has access to.

To simplify the design of customized observation spaces, :code:`gym-anm` allows users to simply specify a set of
variables to include in the observation vectors. For more information on designing new environments, see :ref:`framework_label`.
