..

.. _rewards_label:

Rewards
========
As described in the :ref:`task_overview_label`, the reward signal is computed as:

.. math::
    \begin{align}
        r_t =
        \begin{cases}
            -(\Delta E_{t:t+1} + \lambda \phi(s_{t+1})), & \text{if } s_{t+1} \notin \mathcal S^{terminal}, \\
            - \frac{r^{max}}{1 - \gamma}, & \text{if } s \notin \mathcal S^{terminal} \text{ and }  s_{t+1} \in \mathcal S^{terminal}, \\
            0, & \text{else.}
        \end{cases}
    \end{align}


Energy loss
-----------
The energy loss :math:`\Delta E_{t:t+1}` is computed in three parts:

.. math::
    \begin{align}
        \Delta E_{t:t+1} = \Delta E_{t:t+1}^{(1)} + \Delta E_{t:t+1}^{(2)} + \Delta E_{t:t+1}^{(3)} \;,
    \end{align}

where:

* :math:`\Delta E_{t:t+1}^{(1)}` is the total transmission energy loss during :math:`(t, t+1]`, a result of leakage in
  transmission lines and transformers.
* :math:`\Delta E_{t:t+1}^{(2)}` is the total net amount of energy flowing from the grid into DES units during
  :math:`(t, t+1]`. Over a sufficiently large number of timesteps, the sum of these terms will approximate the amount
  of energy lost due to leakage in DES units.
* :math:`\Delta E_{t:t+1}^{(3)}` is the total amount of energy loss as a result of renewable generation curtailment of
  generators during :math:`(t, t+1]`. Depending on the regulation, this can be thought of as a fee paid by the DNO to
  the owners of the generators that get curtailed, as financial compensation.


Network constraint violation
----------------------------
In the penalty term :math:`\phi(s_{t+1})`, we consider two types of network-wide operating constraints: branch current
limits and voltage constraints (see :ref:`task_overview_label`).

Formally, :math:`\phi(s_{t+1})` is expressed as:

.. math::
    \begin{align}
    \Phi(\mathbf s_{t+1}) = \Delta t \Big(&\sum_{i \in \mathcal N} \big(\max{(0, |V_{i,t+1}| - \overline V_i)} + \max{(0, \underline V_i - |V_{i,t+1}|)} \big) \nonumber \\
    &+ \sum_{e_{ij} \in \mathcal E} \max{(0, |S_{ij,t+1}| - \overline S_{ij}, |S_{ji,t+1}| - \overline S_{ij})} \Big) \;.
    \end{align}

where

* :math:`|V_{i,t+1}|` is the voltage magnitude at bus :math:`i` at time :math:`t+1` (in p.u.),
* :math:`[\underline V_i, \overline V_i]` is the range of allowed voltage magnitude at bus :math:`i` (in p.u.),
* :math:`|S_{ij,t+1}|` is the apparent power flow in branch :math:`e_{ij}` linking buses :math:`i` and :math:`j` at time
  :math:`t+1`,
* :math:`\overline S_{ij}` is the rated (i.e., maximum) apparent power flow of branch :math:`e_{ij}`.

In practice, violating any network constraint can lead to damaging parts of the DN infrastructure (e.g., lines or
transformers) or power outages, which can both have important economical consequences for the DNO. For that reason,
ensuring that the DN operates within its constraints is often prioritized compared to minimizing energy loss. This can
be achieved by choosing a large :math:`\lambda` or by setting an over-restrictive set of constraints in the environment.
