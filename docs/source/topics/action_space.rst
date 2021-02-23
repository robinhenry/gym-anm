..

.. _action_space_label:

Action space
============
Formally, the action vectors :math:`a_t \in \mathcal A` expected by :code:`gym-anm` environments are expressed as:

.. math::
    \begin{align}
        a_t = \big[
        \{a_{P_{g,t}} \}_{g \in \mathcal D_G - \{g^{slack}\}},\; \{a_{Q_{g,t}} \}_{g \in \mathcal D_G - \{g^{slack}\}},
        \{a_{P_{d,t}}\}_{d \in \mathcal D_{DES}},\; \{a_{Q_{d,t}}\}_{d \in \mathcal D_{DES}} \big] \;, \label{eq:action_vector}
    \end{align}

for a total of :math:`N_a = 2|\mathcal D_G| + 2|\mathcal D_{DES}| - 2` control variables to be chosen by the agent at
each timestep, each belonging to one of four categories:

* :math:`a_{P_{g,t}}`: an upper limit on the active power injection from non-slack generator :math:`g`.
  If :math:`g` is a renewable energy resource, then :math:`a_{P_{g,t}}` is the curtailment value. For classical
  generators, it simply refers to a set-point chosen by the agent. The slack generator is excluded, since it is used
  to balance load and generation and therefore its power injection cannot be controlled by the agent. That is,
  :math:`g^{slack}` will inject into the network the amount of power needed to fill the gap between the total
  generation and demand.
* :math:`a_{Q_{g,t}}`: the reactive power injection from each non-slack generator :math:`g`.
  Again, the injection from the slack generator is used to balance reactive power flows and therefore cannot be
  controlled by the agent.
* :math:`a_{P_{d,t}}`: the active power injection from each energy storage unit :math:`d \in \mathcal D_{DES}`.
* :math:`a_{Q_{d,t}}`: the reactive power injection from each energy storage unit :math:`d \in \mathcal D_{DES}`.

As with all Gym environments, the action space :math:`\mathcal A` from which the agent can choose actions can be
obtained by calling :code:`env.action_space()`.

Note that not all actions within :math:`\mathcal A` will be feasible in the current state :math:`s_t` (e.g., an empty
storage unit cannot inject power into the network). The action :math:`a_t \in \mathcal A` chosen by the agent will
first be mapped to the closest action (using Euclidean distance) in the feasible set :math:`\mathcal A(s_t)` before being
applied in the environment.
