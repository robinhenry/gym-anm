..

.. _design_new_env:

Designing New Environments
==========================
The :code:`gym-anm` framework was specifically designed to make it easy for users to design their own
environments and ANM tasks. This page describes in details how to do so.

Template
--------
New environments are created by creating a sub-class of :py:class:`gym_anm.envs.anm_env.ANMEnv`. The general template
to follow is shown below.

.. literalinclude:: ../../../examples/new_env_template.py
   :language: python

where:

* :code:`network` is the network dictionary that describes the characteristics of the power grid
  considered (see Appendix D of `the paper <https://arxiv.org/abs/2103.07932>`_),
* :code:`observation` defines the observation space (see Appendix C of `the paper <https://arxiv.org/abs/2103.07932>`_),
* :code:`K` is the number of auxiliary variables :math:`K`,
* :code:`delta_t` is the time interval (in hour) between subsequent timesteps :math:`\Delta t`,
* :code:`gamma` is the discount factor :math:`\gamma \in [0, 1]`,
* :code:`lamb` is the penalty weighting hyperparameter :math:`\lambda` in the reward function,
* :code:`aux_bounds` are the bounds on the auxiliary variables, specified as a 2D array (column 1 = lower bounds,
  column 2 = upper bounds),
* :code:`costs_clipping` is a tuple of (clip value for :math:`\Delta E_{t:t+1}`, clip value for
  :math:`\lambda \phi(s_{t+1})`), with :math:`r_{clip} = sum(costs\_clipping)`,
* :code:`seed` is a random seed,
* :code:`init_state()` is a method that must be overwritten to return an initial state vector
  :math:`s_0 \sim p_0(\cdot)` (it gets called from :code:`env.reset()`),
* :code:`next_vars()` is a method that must be overwritten to return the next vector of stochastic variables,
  which include (in that order):

  * the active demand :math:`P_{l,t+1}^{(dev)}` of each load :math:`l \in \mathcal D_L` (ordered by their device ID :math:`l`),
  * the maximum generation :math:`P_{g,t+1}^{(max)}` of each non-slack generator :math:`g \in \mathcal D_G - \{g^{slack}\}`
    (ordered by their device ID :math:`g`),
  * the value of each auxiliary variable :math:`aux^{(k)}_{t+1}` for :math:`k=0,\ldots,K-1` (ordered by their auxiliary
    variable ID :math:`k`),

* :code:`observation_bounds()` is an optional method that can be implemented to make the observation
  space finite when :code:`observation` is provided as a callable. In this case, :code:`gym-anm` has no
  way to infer the bounds of observation vectors :math:`o_t` and :code:`observation_bounds()` can be used
  to specify them.
* :code:`render()` and :code:`close()` are optional methods that can be implemented to support rendering
  of the environment. For more information, see the official `Gym <https://gym.openai.com/docs/>`_ documentation.


Example
--------
A concrete example if shown below, where the environment :code:`SimpleEnvironment` is defined for
a 2-bus power grid with a single load connected at bus 1.

.. literalinclude:: ../../../examples/simple_env.py
   :language: python

Notes
-----
Frequent mistakes when designing new :code:`gym-anm` environments include:

1. Failing to specify load power injections as negative injections. This is particularly important in the
   :code:`next_vars()` method, since the demand :math:`P_l^{(dev)}` returned will first get clipped to
   :math:`[\underline P_l, 0]` before being applied to the environment. This means that if a value :math:`>0` is returned,
   it will always get clipped to 0.
