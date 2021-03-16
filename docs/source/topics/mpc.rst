..

.. _mpc_label:

MPC DCOPF policy
================
In the `original paper <https://arxiv.org/abs/2103.07932>`_, we describe a policy based on Model Predictive Control (MPC) that
solves a multi-timestep DC Optimal Power Flow (OPF).

In the paper, we first describe the general MPC-based policy :math:`\pi_{MPC-N}`, which takes as input
predictions of future loads :math:`P_l^{(dev)}` and maximum generator outputs :math:`P_g^{(max)}` over
the optimization horizon :math:`\{t+1,t+N\}`. Two particular cases are then considered:

1. :math:`\pi_{MPC-N}^{constant}`: :math:`P_l^{(dev)}` and :math:`P_g^{(max)}` are assumed constant throughout the optimization horizon,
2. :math:`\pi_{MPC-N}^{perfect}`: :math:`P_l^{(dev)}` and :math:`P_g^{(max)}` are known (i.e., perfectly forecasted).

Constant forecast
-----------------
The first approach can be ran in all :code:`gym-anm` environments. Note, however, that the implementation of
:math:`\pi_{MPC-N}^{constant}` accesses the internal state of the power grid simulator. In other words,
it assumes the environment is fully observable.

A code example is provided below for the environment :code:`ANM6Easy-v0`.

.. literalinclude:: ../../../examples/mpc_constant.py
   :language: python


Perfect forecast
----------------
The second approach can currently only be ran in the environment :code:`ANM6Easy-v0`. This is
because the implementation of :math:`\pi_{MPC-N}^{perfect}` accesses the fixed future loads and
generator outputs stored in the environment object (and custom environments may not have such
fixed time series).

A code example is provided below. Note that the only difference with the example in the
previous section is the change in the class name, from :py:class:`MPCAgent` to :py:class:`MPCAgentANM6Easy`.

.. literalinclude:: ../../../examples/mpc_perfect.py
   :language: python
